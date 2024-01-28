
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from lightning.pytorch import LightningDataModule
from transformers import AutoProcessor, AutoTokenizer
from torch.utils.data import random_split, DataLoader

json_path = "coco_caption_81K.json" #'koala/koala_558K_100K_subset.json'
text_caption_data = json.load(open(json_path))
koala_data = [t for t in text_caption_data if len(t[1]['gpt']) < 120]

class llavadataset(Dataset):
  def __init__(self, img_path, json_path, phi_model_name, clip_model_name):

    self.tokenizer  = AutoTokenizer.from_pretrained(phi_model_name, trust_remote_code=True)
    self.img_path   = img_path
    self.processor  = AutoProcessor.from_pretrained(clip_model_name)
    self.koala_data = koala_data # train_set if train_flag == 'train' else val_set

  def __len__(self):
    return len(self.koala_data)

  def __getitem__(self, idx):
    koala_data = self.koala_data[idx]
    img_name = koala_data[0]
    img = Image.open(self.img_path + '/' + img_name)
    image_processed = self.processor(images=img, return_tensors="pt")['pixel_values']
    image_processed = image_processed.squeeze(0)

    a = self.tokenizer(koala_data[1]['gpt'], return_tensors="pt", return_attention_mask=False)
    return(image_processed , a['input_ids'].squeeze(0))


def collate_fn(batch):
    image_embeddings, captions = zip(*batch)
    image_embeddings_stacked = torch.stack(image_embeddings, dim=0)
    captions_padded = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=50256)
    return (image_embeddings_stacked, captions_padded)