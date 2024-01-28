import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import LayerNorm, Dropout, Linear, Embedding
import lightning.pytorch as pl
from torch.nn import functional as F


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    

class ComplexExpert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ComplexExpert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            #nn.ReLU(),
            NewGELUActivation(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class AdvancedGatingMechanism(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(AdvancedGatingMechanism, self).__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # Average over the sequence length
        x_avg = x.mean(dim=1)
        return self.gate_network(x_avg)

class ComplexMoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super(ComplexMoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([ComplexExpert(input_dim, output_dim) for _ in range(num_experts)])
        self.gating = AdvancedGatingMechanism(input_dim, num_experts)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        gates = self.gating(x)  # [batch_size, num_experts]

        # Expand gates
        gates_expanded = gates.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, num_experts, 1]

        # Stack outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)  # [batch_size, seq_len, num_experts, output_dim]

        # Apply gates
        output = torch.sum(gates_expanded * expert_outputs, dim=2)  # [batch_size, seq_len, output_dim]
        return output



class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super(RotaryEmbedding, self).__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len):
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = t[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)#.to(self.inv_freq.device)
        return torch.sin(emb), torch.cos(emb)



class Attention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention = self.softmax(attention_scores)
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return out
    


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, num_experts):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(input_dim, num_heads)
        self.mixer = ComplexMoE(input_dim, input_dim, num_experts)

    def forward(self, x):
        x = self.mixer(self.attention(x))
        return x
    

class PhiForCausalLM(pl.LightningModule):
    def __init__(self, vocab_size, max_length, input_dim, num_heads, num_experts, num_layers, wandb_logger=None, tokenizer=None):
        super(PhiForCausalLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, input_dim)
        self.rotary_emb = RotaryEmbedding(input_dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(input_dim, num_heads, num_experts) for _ in range(num_layers)])
        self.lm_head = nn.Linear(input_dim, vocab_size)
        self.loss_fct = nn.CrossEntropyLoss()

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token

        self.wandb_logger = wandb_logger

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        sin_emb, cos_emb = self.rotary_emb(x.size(1))
        x = x + sin_emb + cos_emb  # rotary embeddings to input embeddings

        # causal mask
        mask = torch.triu(torch.ones(input_ids.size(1), input_ids.size(1)), diagonal=1).bool()
        #mask = mask.to(input_ids.device)

        for block in self.transformer_blocks:
            x = block(x)
        logits = self.lm_head(x)
        return logits

    def common_step(self, batch):
        input_ids, labels = batch
        logits = self.forward(input_ids)
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log(f"train/loss", loss.item(), prog_bar=True, on_step=True, logger=True) # on_epoch=True,

        if self.global_step % 100 == None:
          context = torch.tensor([818, 262, 3726, 220]).unsqueeze(0)# In the beggining -> 818, 262, 3726, 220
          generated_tokens = self.generate_text(context)
          gen_text = self.tokenizer.decode(generated_tokens[0])
          self.wandb_logger.log_text(key="Generated Text", columns=["Generated Text", "Global Step"], data=[[gen_text, str(self.global_step)]])
          print(gen_text)
          print("===================")

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log(f"val/loss", loss.item(), prog_bar=True, on_step=True, logger=True, sync_dist=True) # on_epoch=True,
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5, betas=(0.9, 0.95),)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=6e-5,
            total_steps=5000,
            pct_start=3/200,
            div_factor=100,
            final_div_factor=10,
            anneal_strategy='cos'
        )
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
      optimizer.zero_grad(set_to_none=True)

    def generate_text(self, context, max_new_tokens=50):
      for _ in range(max_new_tokens):
        idx_crop = context[:, -max_new_tokens:]
        logits = self(idx_crop)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, idx_next), dim=-1)
      return(context)