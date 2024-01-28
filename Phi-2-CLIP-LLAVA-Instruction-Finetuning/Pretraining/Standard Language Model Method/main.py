import glob2
import torch
import lightning.pytorch as pl
from network import CLIPPhi2Model, clip_model_name, phi_model_name
from dataset import llavadataset, collate_fn
from torch.optim.lr_scheduler import OneCycleLR
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import random_split, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor


max_steps = 13000
num_workers = 32

checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="train/loss",
    mode="min",
    dirpath="model-checkpoint/",
    filename="MModalGPT-{step:02d}-{train/loss:.2f}",
    every_n_train_steps=100
)


def main():
    img_path   = 'instruct150K'
    batch_size = 2
    caption_path = 'coco_caption_81K.json'

    train_dataloader = DataLoader(llavadataset(img_path, caption_path, phi_model_name, clip_model_name),
                                  collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers, 
                                  shuffle=True, pin_memory=True)

    wandb_logger = WandbLogger(project="MLM-Phi2-CLIP-Pretraining", name='standard-coco-MOE4', log_model=True)
    MModalGPT = CLIPPhi2Model()
    trainer = pl.Trainer(
                            log_every_n_steps=1,
                            enable_model_summary=True,
                            max_steps=max_steps,
                            accelerator='auto',
                            devices='auto',
                            logger=[wandb_logger],
                            callbacks=[checkpoint_callback,
                                       LearningRateMonitor(logging_interval="step")]
                        )
    
    torch.set_float32_matmul_precision('medium')
    trainer.fit(MModalGPT, train_dataloader)

if __name__ == "__main__":
    main()