# evaluate.py
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from transformers import Trainer, TrainingArguments
from backend.utils.data_cleaning import clean_text
from model import BERTModelDataset, initialize_model, get_tokenizer, compute_metrics

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved transformer model on a labeled CSV dataset"
    )
    parser.add_argument(
        "--model_dir", "-m", type=str, required=True,
        help="Path to the directory containing the saved model & tokenizer"
    )
    parser.add_argument(
        "--data_path", "-d", type=str, required=True,
        help="Path to the CSV file for evaluation"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=16,
        help="Batch size for evaluation (default: 16)"
    )
    parser.add_argument(
        "--max_len", "-l", type=int, default=124,
        help="Max token length (must match training)"
    )
    args = parser.parse_args()

    # 1. Load and preprocess data
    df = pd.read_csv(args.data_path)
    if 'Tweet_id' in df.columns:
        df = df.drop(columns=['Tweet_id'])
    df['clean_text'] = df['Text'].astype(str).apply(clean_text)

    # 2. Prepare label map
    labels = sorted(df['sentiment'].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}

    # 3. Create dataset
    eval_dataset = BERTModelDataset(
        texts=df['clean_text'].tolist(),
        labels=df['sentiment'].tolist(),
        tokenizer_name=args.model_dir,
        max_len=args.max_len,
        label_map=label_map
    )

    # 4. Load model & tokenizer
    model = initialize_model(args.model_dir, num_labels=len(label_map))
    tokenizer = get_tokenizer(args.model_dir)

    # 5. Set up Trainer for evaluation
    eval_args = TrainingArguments(
        output_dir="eval_output",
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=eval_args,
        compute_metrics=compute_metrics
    )

    # 6. Run evaluation
    results = trainer.evaluate(eval_dataset=eval_dataset)
    print("\n🔎 Evaluation Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
