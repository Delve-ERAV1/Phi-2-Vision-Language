import torch
from multiprocessing import Pool
from typing import List
from torch.utils.data import TensorDataset, DataLoader, random_split
from lightning import LightningDataModule


def process_file(file_path, tokenizer, max_length):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        text = file.read()
    return tokenizer.encode(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')

def process_files_in_parallel(file_paths: List[str], tokenizer, max_length: int) -> torch.Tensor:
    with Pool() as pool:
        results = pool.starmap(process_file, [(file_path, tokenizer, max_length) for file_path in file_paths])

    # Concatenate results and return
    input_ids = torch.cat(results, dim=0)
    return input_ids


class TextDatasetDataModule(LightningDataModule):
    def __init__(self, input_ids: torch.Tensor, batch_size: int, split_ratio: float = 0.9):
        super().__init__()
        self.input_ids = input_ids
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # shift input ids
        target_ids = self.input_ids.clone()
        target_ids[:, :-1] = self.input_ids[:, 1:]
        dataset = TensorDataset(self.input_ids, target_ids)

        # Split
        train_size = int(self.split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
