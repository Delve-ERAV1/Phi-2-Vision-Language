import torch
from PIL import Image
import torch.nn as nn
from torch.nn import functional as F
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoProcessor, AutoTokenizer
from dataset import phi_model_name, tokenizer
from transformers import CLIPVisionModel, AutoModelForCausalLM



class CLIPPhi2Model(LightningModule):
    def __init__(self, phi_model_name, clip_embed=768, phi_embed=2560):
        super().__init__()
        self.EOS_TOKEN_ID = 50256        
        self.text_model = AutoModelForCausalLM.from_pretrained(phi_model_name,
                                                               torch_dtype=torch.float16,
                                                               device_map="cuda", 
                                                               trust_remote_code=True)
        self.projection = Projections(clip_embed, phi_embed)
        self.tokenizer = tokenizer

        # Freeze Weights
        for param in self.text_model.parameters():
            param.requires_grad_(False)

    def forward(self, images, input_ids):
        input_embeddings = self.text_model.model.embed_tokens(input_ids)
        projected_image_embeds = self.projection(images).to(torch.float16)
        combined_embeddings = torch.cat((projected_image_embeds, input_embeddings), dim=1)
        outputs = self.text_model(inputs_embeds=combined_embeddings).logits

        del combined_embeddings
        del input_embeddings
        return outputs#[:, -1, :]


    def training_step(self, batch, batch_idx):
        images, input_ids, target_ids = batch
        outputs = self.forward(images, input_ids)

        # Select the logits for all text tokens after the 5 separator tokens
        text_token_logits = outputs[:, 54:, :]  # Start from index 5 to skip separator tokens

        # Construct the target sequence for text tokens, including the next token
        # Skip the first 5 separator tokens
        target_sequence = torch.cat([input_ids[:, 5+1:], target_ids], dim=1)

        # Flatten the logits and target sequence for loss calculation
        text_token_logits_flat = text_token_logits.reshape(-1, text_token_logits.size(-1))
        target_sequence_flat = target_sequence.reshape(-1)

        # Calculate loss over the text token sequence
        loss = F.cross_entropy(text_token_logits_flat, target_sequence_flat, ignore_index=self.EOS_TOKEN_ID)

        self.print_predictions(batch, self.global_step)
        self.log(f"loss", loss.item(), prog_bar=True, on_step=True, logger=True)
        return loss



    def print_predictions(self, batch, global_step):
        if global_step % 100 == 0:
            images, input_ids, target_ids = batch

            # Generate predictions and select the first 4 samples in the batch
            outputs = self.forward(images, input_ids)
            predicted_token_ids = outputs.argmax(dim=-1)[:4]

            # Construct full target sequences excluding separator tokens
            # Assuming the total length of image and separator tokens is 54 (49 + 5)
            full_target_sequences = [torch.cat((input_seq[5:], target_seq), dim=0) for input_seq, target_seq in zip(input_ids[:4], target_ids[:4])]

            # Convert token IDs to text
            predicted_captions = [self.tokenizer.decode(ids[ids != self.EOS_TOKEN_ID][50:]) for ids in predicted_token_ids]
            target_captions = [self.tokenizer.decode(ids[ids != self.EOS_TOKEN_ID]) for ids in full_target_sequences]

            # Print target and predicted captions
            for i in range(4):

                print('predicted_captions-', len(predicted_captions[i]))
                print('target_captions ', len(target_captions[i]))
                print(f"Image {i+1}:")
                print(f"Target Caption: {target_captions[i]}")
                print(f"Predicted Caption: {predicted_captions[i]}")
                print("------------")



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-4,
            pct_start=0.01,
            total_steps=self.trainer.max_steps,
            anneal_strategy='cos',
            div_factor=100,
            final_div_factor=1000,
        )

        return {'optimizer': optimizer, 
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def on_save_checkpoint(self, checkpoint):

        keys = checkpoint['state_dict'].keys()
        keys = [k for k in keys if 'projection' not in k]

        for k in keys:
            del checkpoint['state_dict'][k]
    

class Projections(nn.Module):
    def __init__(
        self,
        clip_embed,
        phi_embed,
        num_experts=2,
        num_projection_layers=6,
    ):
        super().__init__()

        #self.MixtureOfExperts = MixtureOfExperts(clip_embed, num_experts)
        self.norm = nn.LayerNorm(phi_embed)
        self.output = nn.Linear(clip_embed, phi_embed)
        self.projection_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(phi_embed, phi_embed),
                    nn.GELU(),  
                    nn.Linear(phi_embed, phi_embed),
                )
                for _ in range(num_projection_layers)
            ]
        )

    def forward(self, x):
        #x = self.MixtureOfExperts(x)
        x = self.output(x)
        self.norm(x)
        for layer in self.projection_layers:
            residual = x
            x = layer(x) + residual 
        
        return x


class MixtureOfExperts(nn.Module):
    def __init__(self, embed, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(embed, embed) for _ in range(num_experts)])
        self.gating = nn.Linear(embed, num_experts)

    def forward(self, x):
        gates = torch.softmax(self.gating(x), dim=-1).unsqueeze(-1) 
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        output = torch.sum(gates*expert_outputs, dim=2)
        return output