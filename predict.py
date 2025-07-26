# predict.py
import pandas as pd
import os
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from last_train import data_cleaning, BERTModelDataset  # Import from training file

# Constants
Tweets_Ids_Col_Test = "Tweet_id"
Tweets_Text_Col_Test = "Text"
Test_Data_File = os.getenv('TEST_DATA_FILE', 'Sentiment Analysis of ARABIC Tweets/test1_with_text.csv')
Output_File = os.getenv('OUTPUT_FILE', 'Sentiment Analysis of ARABIC Tweets/test.csv')
Model_Path = os.getenv('MODEL_PATH', 'Sentiment Analysis of ARABIC Tweets/saved_model')  # Path to saved model from training
Max_Len = 124  # Should match what was used in training
label_list = ['neutral', 'negative', 'positive']  # Should match training order


class Predictor:
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERTv2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize trainer for consistent behavior
        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="./tmp"),
        )

    def predict(self, text):
        encoded_review = self.tokenizer.encode_plus(
            text,
            max_length=Max_Len,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            truncation='longest_first',
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoded_review['input_ids'].to(self.device)
        attention_mask = encoded_review['attention_mask'].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
            _, prediction = torch.max(output[0], dim=1)

        return prediction[0]


def main():
    # Load test data
    test_data = pd.read_csv(Test_Data_File, sep=",")
    test_data.columns = [Tweets_Ids_Col_Test, Tweets_Text_Col_Test]
    test_data[Tweets_Text_Col_Test] = test_data[Tweets_Text_Col_Test].apply(lambda x: data_cleaning(x))

    # Initialize predictor
    predictor = Predictor(Model_Path)

    # Make predictions
    prediction_list = []
    for i, tweet in enumerate(test_data[Tweets_Text_Col_Test]):
        tweet_id = test_data[Tweets_Ids_Col_Test][i]
        prediction = predictor.predict(tweet)
        pred_txt = label_list[prediction]

        # Convert to numerical labels
        if pred_txt == 'positive':
            pred_txt = 1
        elif pred_txt == 'negative':
            pred_txt = -1
        else:
            pred_txt = 0

        prediction_list.append(pred_txt)

    # Save results
    results = pd.DataFrame({
        'Tweet_id': test_data[Tweets_Ids_Col_Test].astype(str),
        'sentiment': prediction_list
    })

    results.to_csv(Output_File, sep=",", index=False)
    print(f"Predictions saved to {Output_File}")


if __name__ == "__main__":
    main()