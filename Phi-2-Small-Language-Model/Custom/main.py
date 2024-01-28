import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import glob2
from dataset import TextDatasetDataModule, process_files_in_parallel
import lightning.pytorch as pl
from network import PhiForCausalLM, TransformerBlock
from transformers import AutoTokenizer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor


text_files = glob2.glob("Project_Gutenberg/*.txt")
MAX_LENGTH = 128  # Maximum sequence length
BATCH_SIZE = 32    # Batch size
DATA_SPLIT = 0.9

def main():
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = process_files_in_parallel(text_files[:100], tokenizer, MAX_LENGTH)
    data_module = TextDatasetDataModule(input_ids, batch_size=BATCH_SIZE)
    wandb_logger = WandbLogger(project="Phi2-Scratch-pretraining", name='sample-small')
    
    model = PhiForCausalLM(vocab_size=51200, max_length=MAX_LENGTH, input_dim=2560, 
                           num_heads=8, num_experts=2, num_layers=2,
                           wandb_logger=wandb_logger, tokenizer=tokenizer)

    strategy = FSDPStrategy(
        auto_wrap_policy={TransformerBlock},
        activation_checkpointing_policy={TransformerBlock},
        state_dict_type="full",
        limit_all_gathers=True,
        cpu_offload=False,
    )
    trainer = pl.Trainer(
                        log_every_n_steps=1,
                        check_val_every_n_epoch=10,
                        enable_model_summary=True,
                        max_steps=5000,
                        accelerator='auto',
                        devices=4,
                        strategy=strategy,
                        precision='16',
                        logger=[wandb_logger],
                        callbacks=[LearningRateMonitor(logging_interval="step")]
                        )

    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()

# model = PhiForCausalLM(vocab_size=51200, max_length=1024, input_dim=2560, 
#                            num_heads=8, num_experts=4, num_layers=32,
#                            context=context, wandb_logger=wandb_logger, tokenizer=tokenizer)