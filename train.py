# train.py
# -*- coding: utf-8 -*-

import os
import argparse
import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer, TrainingArguments, EarlyStoppingCallback,
    DataCollatorWithPadding
)

from backend.utils.data_cleaning import clean_text
from model import BERTModelDataset, initialize_model, get_tokenizer, compute_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train a transformer for Arabic tweet sentiment classification"
    )
    parser.add_argument("--data_path", "-d", type=str, default="data/train_small_frac10.csv",
                        help="Path to the CSV dataset (sampled or full)")
    parser.add_argument("--output_dir", "-o", type=str, default="./model_out",
                        help="Directory to save the fine-tuned model and tokenizer")
    parser.add_argument("--max_len", type=int, default=124,
                        help="Maximum token length")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Prompt for model name
    print("Enter a HuggingFace model ID (e.g. UBC-NLP/MARBERTv2, asafaya/bert-base-arabic):")
    model_name = input("Model ID: ").strip()

    # Load and preprocess data
    df = pd.read_csv(args.data_path)
    if 'Tweet_id' in df.columns:
        df = df.drop(columns=['Tweet_id'])
    df['clean_text'] = df['Text'].astype(str).apply(clean_text)

    # Split into train / eval
    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.seed,
        stratify=df['sentiment']
    )

    # Label mapping
    labels = sorted(train_df['sentiment'].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}

    # Create dataset objects
    train_dataset = BERTModelDataset(
        texts=train_df['clean_text'].tolist(),
        labels=train_df['sentiment'].tolist(),
        tokenizer_name=model_name,
        max_len=args.max_len,
        label_map=label_map
    )
    eval_dataset = BERTModelDataset(
        texts=eval_df['clean_text'].tolist(),
        labels=eval_df['sentiment'].tolist(),
        tokenizer_name=model_name,
        max_len=args.max_len,
        label_map=label_map
    )

    # Initialize model & tokenizer
    model = initialize_model(model_name, num_labels=len(label_map))
    tokenizer = get_tokenizer(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training settings (💡 disables progress bar & reduces logs)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=500,         # ✅ Log less frequently
        disable_tqdm=True,         # ✅ Remove tqdm bars
        seed=args.seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # Set seed and train
    set_seed(args.seed)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    print(f"\n▶️ Starting training with `{model_name}`...\n")
    trainer.train()

    # Save final model and tokenizer in PyTorch .bin format
    model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n✅ Training complete. Final model saved to `{args.output_dir}`")


if __name__ == "__main__":
    main()
