
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
from torch.utils.data import random_split, DataLoader


phi_model_name = "microsoft/phi-2"
image_path = "instruct150K_embeddings/"
tokenizer  = AutoTokenizer.from_pretrained(phi_model_name, trust_remote_code=True)


class ImageCaptionDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list of tuples): List of tuples containing (image_name, input_ids, target_id).
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, input_ids, target_id = self.data[idx]
        image_name = torch.load(image_path+image_name)
        return image_name.squeeze(0), input_ids, target_id
    

def collate_fn(batch):
    image_names, input_ids, target_ids = zip(*batch)

    images = torch.stack(image_names, dim=0)
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(ids) for ids in input_ids],
         batch_first=True,
         padding_value=tokenizer.eos_token_id
    )

    target_ids_tensor = torch.tensor(target_ids)

    return images, input_ids_padded, target_ids_tensor