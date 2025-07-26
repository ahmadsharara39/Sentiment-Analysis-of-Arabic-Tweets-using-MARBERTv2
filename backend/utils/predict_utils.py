import pandas as pd
import torch
from utils.model_loader import load_model_and_tokenizer
from utils.data_cleaning import clean_text


def predict_from_csv(csv_path, model_name):
    df = pd.read_csv(csv_path)
    df.columns = [col.lower() for col in df.columns]

    # Flexible column detection
    text_col = next((col for col in ['text', 'tweet', 'content'] if col in df.columns), None)
    if not text_col:
        raise ValueError("❌ Uploaded CSV must contain 'text', 'tweet', or 'content'.")

    # Clean
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].str.strip() != '']

    df['cleaned_text'] = df[text_col].apply(clean_text)

    tokenizer, model = load_model_and_tokenizer(f"outputs/{model_name}")
    model.eval()

    results = []
    for text in df['cleaned_text']:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            label = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()
            results.append({
                "text": text,
                "label": label,
                "confidence": confidence
            })

    return results, df


def predict_single_tweet(text, model_name):
    text = clean_text(text)
    tokenizer, model = load_model_and_tokenizer(f"outputs/{model_name}")
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = torch.max(probs).item()
        label = torch.argmax(probs, dim=1).item()

    return label, confidence
