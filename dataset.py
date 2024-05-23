# dataset.py
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

    def __len__(self):
        return len(self.labels)

def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            text, label = line.strip().split('\t')
            texts.append(text)
            labels.append(int(label))
    return texts, labels
