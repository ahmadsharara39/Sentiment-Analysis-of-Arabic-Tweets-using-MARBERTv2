from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer(model_path):
    """
    Loads a model and tokenizer from the given path using Hugging Face's Auto classes.
    
    Parameters:
    - model_path (str): Path to the directory containing the saved model and tokenizer.

    Returns:
    - tokenizer: Hugging Face tokenizer
    - model: Hugging Face model (AutoModelForSequenceClassification)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model
