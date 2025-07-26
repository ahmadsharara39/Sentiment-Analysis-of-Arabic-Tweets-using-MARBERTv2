# ensemble.py
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.utils.data_cleaning import clean_text

def resolve_checkpoint_dir(model_dir: str) -> str:
    """
    Given a model directory, find the actual checkpoint directory:
      1. If `model_dir` itself contains config.json or a known weight file, return it.
      2. Else if it contains subfolders checkpoint-*, pick the highest-numbered one.
      3. Else if it contains exactly one subfolder, recurse into that.
    """
    # 1) check root for config.json or weight file
    for fname in ("config.json",
                  "pytorch_model.bin",
                  "model.safetensors",
                  "tf_model.h5",
                  "model.ckpt.index"):
        if os.path.isfile(os.path.join(model_dir, fname)):
            return model_dir

    # 2) look for checkpoint-* subdirs
    ckpts = [
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d)) and d.startswith("checkpoint-")
    ]
    if ckpts:
        ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))
        return os.path.join(model_dir, ckpts_sorted[-1])

    # 3) if there's exactly one nested folder, descend into it
    subs = [
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    ]
    if len(subs) == 1:
        return resolve_checkpoint_dir(os.path.join(model_dir, subs[0]))

    raise ValueError(
        f"No config or weights found in '{model_dir}', "
        f"and no unambiguous checkpoint subfolder."
    )

def load_model(model_dir: str, device: torch.device):
    real_dir = resolve_checkpoint_dir(model_dir)
    print(f"→ Loading model from: {real_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(
        real_dir,
        local_files_only=True
    )
    model.to(device)
    model.eval()
    return model

def predict_with_model(model, tokenizer, texts, max_len, device):
    logits_list = []
    for text in texts:
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
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits_list.append(outputs.logits.cpu().squeeze(0))
    return torch.stack(logits_list)

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble multiple fine-tuned models by averaging logits"
    )
    parser.add_argument(
        "-m", "--model_dirs", nargs="+", required=True,
        help="Paths to each saved model directory"
    )
    parser.add_argument(
        "-i", "--input_csv", required=True,
        help="CSV file with columns ['Tweet_id','Text']"
    )
    parser.add_argument(
        "-o", "--output_csv", default="ensemble_predictions.csv",
        help="Where to write out ['Tweet_id','sentiment']"
    )
    parser.add_argument(
        "--max_len", type=int, default=124,
        help="Max token length (must match training)"
    )
    args = parser.parse_args()

    # 1. Load data
    df = pd.read_csv(args.input_csv)
    if 'Tweet_id' not in df.columns or 'Text' not in df.columns:
        raise ValueError("Input CSV must have 'Tweet_id' and 'Text' columns")

    texts = df['Text'].astype(str).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load each model & tokenizer
    models, tokenizers = [], []
    for path in args.model_dirs:
        real_dir = resolve_checkpoint_dir(path)
        models.append(load_model(path, device))
        tokenizers.append(
            AutoTokenizer.from_pretrained(real_dir, local_files_only=True)
        )

    # 3. Sum logits
    summed_logits = None
    for model, tokenizer in zip(models, tokenizers):
        logits = predict_with_model(model, tokenizer, texts, args.max_len, device)
        summed_logits = logits if summed_logits is None else summed_logits + logits

    # 4. Average and predict
    avg_logits = summed_logits / len(models)
    preds = torch.argmax(avg_logits, dim=1).tolist()

    # 5. Save results
    out = pd.DataFrame({
        "Tweet_id": df['Tweet_id'],
        "sentiment": preds
    })
    out.to_csv(args.output_csv, index=False)
    print(f"✅ Ensemble predictions saved to {args.output_csv}")

if __name__ == "__main__":
    main()
