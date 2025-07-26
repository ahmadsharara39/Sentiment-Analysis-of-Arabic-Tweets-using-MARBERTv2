# model.py
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

class BERTModelDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenized inputs and labels for transformer-based classification.
    """
    def __init__(self, texts, labels, tokenizer_name: str, max_len: int, label_map: dict):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.label_map = label_map

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).strip()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels':         torch.tensor(self.label_map[self.labels[idx]], dtype=torch.long)
        }

def initialize_model(model_name: str, num_labels: int, force_download: bool = False):
    """
    Load a pretrained transformer model for sequence classification.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        force_download=force_download
    )

def get_tokenizer(model_name: str):
    """
    Return the tokenizer corresponding to the given model name.
    """
    return AutoTokenizer.from_pretrained(model_name)

def compute_metrics(pred):
    """
    Compute classification metrics: accuracy, macro F1, precision, recall.
    """
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    print(classification_report(labels, preds))
    return {
        'accuracy':        accuracy_score(labels, preds),
        'macro_f1':        f1_score(labels, preds, average='macro'),
        'macro_precision': precision_score(labels, preds, average='macro'),
        'macro_recall':    recall_score(labels, preds, average='macro')
    }
