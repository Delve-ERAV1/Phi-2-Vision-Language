import json
import torch
import lightning.pytorch as pl
from network import CLIPPhi2Model
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from dataset import ImageCaptionDataset, collate_fn, phi_model_name



checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="loss",
    mode="min",
    dirpath="model-checkpoint/",
    filename="MModalGPT-{step:02d}-{loss:.2f}",
    every_n_train_steps=100
)


def main():
    batch_size = 32 
    num_workers = 16
    max_steps = 20000
    wandb_logger = WandbLogger(project="MLM-Phi2-CLIP-Pretraining-coco", name='exp', log_model=True)

    data = json.load(open('autoregress_finetune.json'))
    dataset = sorted(data, key=lambda x: len(x[1]))

    train_dataset = ImageCaptionDataset(dataset)
    train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_fn,
                        num_workers=num_workers)


    MModalGPT = CLIPPhi2Model(phi_model_name)
    trainer = pl.Trainer(
                log_every_n_steps=1,
                enable_model_summary=True,
                max_steps=max_steps,
                accelerator='auto',
                devices='auto',
                strategy='ddp_find_unused_parameters_true',
                logger=[wandb_logger],
                callbacks=[checkpoint_callback,
                           LearningRateMonitor(logging_interval="step")], 
                )
    
    torch.set_float32_matmul_precision('medium')
    trainer.fit(MModalGPT, train_dataloader)


if __name__ == "__main__":
    main()