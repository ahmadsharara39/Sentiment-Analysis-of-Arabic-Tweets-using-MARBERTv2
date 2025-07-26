# new_predict.py
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.utils.data_cleaning import clean_text

def load_model_and_tokenizer(model_dir: str, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer

def predict_single(model, tokenizer, text: str, max_len: int, device: torch.device) -> int:
    cleaned = clean_text(text)
    inputs = tokenizer.encode_plus(
        cleaned,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return torch.argmax(logits, dim=-1).item()

def main():
    parser = argparse.ArgumentParser(description="Run sentiment predictions on a CSV of Arabic tweets")
    parser.add_argument("-m","--model_dir", required=True, help="Directory with saved model/tokenizer")
    parser.add_argument("-i","--input_csv", required=True, help="Input CSV file")
    parser.add_argument("-o","--output_csv", default="predictions.csv", help="Output CSV file")
    parser.add_argument("--id_col", default="Tweet_id",
                        help="Identifier column name (default: Tweet_id)")
    parser.add_argument("--text_col", default="Text",
                        help="Text column name (default: Text)")
    parser.add_argument("--headerless", action="store_true",
                        help="Set if the CSV has no header row (just two columns)")
    parser.add_argument("--max_len", type=int, default=124,
                        help="Max token length (must match training)")
    args = parser.parse_args()

    # Load DataFrame, with or without header
    if args.headerless:
        df = pd.read_csv(args.input_csv, header=None, names=[args.id_col, args.text_col])
    else:
        df = pd.read_csv(args.input_csv)

    # Check columns
    for col in (args.id_col, args.text_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {args.input_csv}. Available: {df.columns.tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)

    preds = [predict_single(model, tokenizer, t, args.max_len, device)
             for t in df[args.text_col].astype(str)]

    out = pd.DataFrame({args.id_col: df[args.id_col], "sentiment": preds})
    out.to_csv(args.output_csv, index=False)
    print(f"✅ Predictions saved to {args.output_csv}")

if __name__ == "__main__":
    main()
