import peft
import torch
import pathlib
import torch.nn as nn
from peft import LoraConfig
from dataset import tokenizer
from torch.nn import functional as F
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoModelForCausalLM, BitsAndBytesConfig



class CLIPQuestionAnsweringModel(LightningModule):
    def __init__(self, phi_model_name, clip_embed=768, phi_embed=2560):
        super().__init__()
        self.EOS_TOKEN_ID = 50256
        self.QUESTION_ANSWER_SEPARATOR_ID = 50295  # Special token ID for question-answer separation
        self.IMAGE_SEPARATOR_TOKENS = [685, 36259, 14041, 60, 220]

        self.projection = load_projection_model("MModalGPT-step=13800-loss=0.39.ckpt", clip_embed, phi_embed)
        self.tokenizer = tokenizer
        
        self.bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                        )
        self.text_model = AutoModelForCausalLM.from_pretrained(phi_model_name,
                                                               # torch_dtype=torch.float16,
                                                               device_map="cuda",
                                                               quantization_config=self.bnb_config,
                                                               trust_remote_code=True)
        self.text_model.config.use_cache = False
        self.peft_config = LoraConfig(
                                lora_alpha=16, lora_dropout=0.1, r=64,
                                bias="none", task_type="CAUSAL_LM",
                                target_modules=[
                                    "q_proj",
                                    'k_proj',
                                    'v_proj',
                                    'fc1',
                                    'fc2'])
        
        self.peft_model = peft.get_peft_model(self.text_model, self.peft_config)
        

    def forward(self, images, input_ids):

        input_embeddings = self.peft_model.model.model.embed_tokens(input_ids)
        projected_image_embeds = self.projection(images).to(torch.float16)
        combined_embeddings = torch.cat((projected_image_embeds, input_embeddings), dim=1)
        outputs = self.peft_model(inputs_embeds=combined_embeddings).logits
        del combined_embeddings, input_embeddings
        return outputs

    def training_step(self, batch, batch_idx):
        images, input_ids, target_ids = batch
        outputs = self.forward(images, input_ids)
    
        separator_indices = (input_ids == self.QUESTION_ANSWER_SEPARATOR_ID).nonzero(as_tuple=True)[1]
    
        answer_start_indices = separator_indices + 1
    
        collected_logits = []
        collected_targets = []
    
        for i in range(input_ids.size(0)):

            if answer_start_indices[i] + 49 < outputs.size(1):
                collected_logits.append(outputs[i, answer_start_indices[i] + 49, :])
                collected_targets.append(target_ids[i])

            elif answer_start_indices[i] < outputs.size(1):
                collected_logits.append(outputs[i, -1, :])
                collected_targets.append(target_ids[i])
    
        answer_logits_flat = torch.cat(collected_logits).reshape(-1, outputs.size(-1))
        target_sequences_flat = torch.cat(collected_targets)
    
        loss = F.cross_entropy(answer_logits_flat, target_sequences_flat, ignore_index=self.EOS_TOKEN_ID)
    
        self.log("loss", loss, prog_bar=True, on_step=True, logger=True)

        self.print_predictions(batch, self.global_step)

        del outputs, answer_logits_flat, collected_logits
        return loss

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-11)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-6,
            pct_start=0.01,
            total_steps=self.trainer.max_steps,
            anneal_strategy='cos',
            div_factor=100,
            final_div_factor=1000,
        )
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
    
                
    def print_predictions(self, batch, global_step):
        if global_step % 100 == 0:  # Print every 100 steps
            images, input_ids, target_ids = batch
            num_examples = 4
    
            outputs = self.forward(images, input_ids)
            predicted_token_ids = outputs.argmax(dim=-1)[:num_examples]
    
            separator_indices = (input_ids == self.QUESTION_ANSWER_SEPARATOR_ID).nonzero(as_tuple=True)[1][:num_examples]
            image_separator_indices = (input_ids == self.IMAGE_SEPARATOR_TOKENS[0]).nonzero(as_tuple=True)[1][:num_examples]
    
            predicted_answers = [self.tokenizer.decode(ids[(sep_index + 49):].tolist(), skip_special_tokens=True) for ids, sep_index in zip(predicted_token_ids, separator_indices)]
            actual_answers = [self.tokenizer.decode(ids[(sep_index + 1):].tolist(), skip_special_tokens=True) for ids, sep_index in zip(input_ids[:num_examples], separator_indices)]
    
            questions = [self.tokenizer.decode(input_ids[i, image_sep_index+1:q_sep_index].tolist()) for i, (q_sep_index, image_sep_index) in enumerate(zip(separator_indices, image_separator_indices))]
    
            for i in range(num_examples):
                print(f"Sample {i+1}:")
                print(f"Question: {questions[i][15:]}")
                print(f"Predicted Answer: {predicted_answers[i]}")
                print(f"Actual Answer: {actual_answers[i]}")
                print("------------")


    def on_save_checkpoint(self, checkpoint):
        path_location = f"peft-checkpoint/{self.global_step}"
        path = pathlib.Path(path_location)
        path.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(path)

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
    


def load_projection_model(path, clip_embed, phi_embed):
    """Loads a Projections model instance from a checkpoint and returns it with weights loaded.

    Args:
        path (str): Path to the checkpoint file.

    Returns:
        torch.nn.Module: The loaded Projections model instance.
    """

    state_dict = torch.load(path)['state_dict']
    new_state_dict = {k.replace('projection.', ''): v for k, v in state_dict.items()}

    model = Projections(clip_embed, phi_embed)  
    model.load_state_dict(new_state_dict)

    return model
