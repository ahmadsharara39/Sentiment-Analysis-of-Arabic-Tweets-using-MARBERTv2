<!-- README.md -->

# Sentiment Analysis of Arabic Tweets

A modular pipeline for Arabic‐tweet sentiment classification with multiple transformer backbones and a logits‐averaging ensemble.

## 🔧 Setup

1. **Clone** or copy this folder locally.  
2. **Install** dependencies:
   ```bash
   pip install -r requirements.txt

3. **Train**:
python train.py \
  --data_path train_all_ext.csv \
  --output_dir outputs/marbertv2_run \
  --max_len 124 \
  --batch_size 16 \
  --epochs 2 \
  --seed 42
when prompted type: UBC-NLP/MARBERTv2
Repeat for asafaya/bert-base-arabic → outputs/asafaya_run,
and aubmindlab/bert-base-arabertv2 → outputs/arabertv2_run.

4. **Evaluate**: 
python evaluate.py \
  --model_dir outputs/marbertv2_run \
  --data_path Sampled_Dataset.csv \
  --batch_size 16 \
  --max_len 124
we can evaluate using any model with any dataset we have

5. **Ensemble**: 
python ensemble.py \
  --model_dirs \
    outputs/marbertv2_run \
    outputs/asafaya_run \
    outputs/arabertv2_run \
  --input_csv train_all_ext.csv \
  --output_csv ensemble_predictions.csv \
  --max_len 124

Ensemble Strategy
This document explains the design and rationale behind our ensemble.py logits‐averaging approach.

1. What the script does
  1. Resolves each checkpoint
      Handles flat folders or nested checkpoint-<n> directories.
      Picks the latest checkpoint automatically.

  2. Loads each model & tokenizer (using local_files_only=True to avoid any Hugging Face Hub calls).

  3. Cleans and tokenizes each tweet via the shared data_cleaning.clean_text().

  4. Runs a forward pass to collect each model’s raw logits (the vector of pre‐softmax scores for each class).

  5. Sums all logits together, averages by the number of models, then takes an argmax over classes.

  6. Writes out a final CSV of Tweet_id → ensembled class index.

2. Why average logits (not probabilities or hard votes)?
| Aspect                   | Hard‐voting classes | Averaging probs                | Averaging logits        |
| ------------------------ | ------------------- | ------------------------------ | ----------------------- |
| **Preserves confidence** | No                  | Partially (bounded 0–1)        | Yes (unbounded scores)  |
| **Numerical stability**  | N/A                 | Needs softmax → underflow risk | Avoids repeated softmax |
| **Linearly comb’ble**    | No                  | Yes                            | Yes                     |
| **Calibration**          | Poor                | Depends                        | Often better            |

Logits are linear, unbounded model outputs. Averaging them directly:
  Preserves relative confidence gaps between classes.
  Avoids numerical under‐/overflow that can occur when you softmax very large or small values.
  Eliminates one extra softmax pass per model, saving compute.
After averaging logits, a single argmax picks the final class.

3. Why not use an off-the-shelf ensembling library?
  Zero extra dependencies: We already depend on PyTorch + Transformers. No need to pull in scikit-learn’s VotingClassifier or external packages.
  Full transparency: The entire logic (checkpoint resolution, tokenization, logit aggregation) lives in one concise, easily‐auditable script.
  Customizability:
    You can drop in weighted averaging simply by replacing
      summed_logits += logits with summed_logits += weight * logits  
    You can swap to majority‐vote on torch.argmax(logits,1) if desired.
    You can feed the averaged logits into a small meta‐classifier (“stacking”) if you wish.

4. Extensions & Alternatives
  Weighted ensembling: assign higher weights to stronger models (e.g. MarBERTv2 → 0.4, AraBERTv2 → 0.3, AraBERT → 0.3).
  Calibrated probabilities: if you care about well‐formed probabilities, you can softmax the averaged logits or apply temperature scaling.
  Meta‐learning / stacking: collect each model’s logits as features, then train a simple logistic‐regressor on a held‐out validation set.

  🌐 Running the Website
Follow these steps to run the Sentiment Analysis of Arabic Tweets website locally.

1. Frontend Setup (React App)
The frontend is built with React. It allows you to interact with the backend, input tweets, and view sentiment predictions.

a. Navigate to the frontend directory:
Open a terminal (or PowerShell) and navigate to the frontend directory in your project:
cd 'Sentiment Analysis of ARABIC Tweets\frontend'

b. Allow script execution (only required once on Windows):
Run this command to allow scripts to execute:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

c. Install dependencies:
Install the required npm packages by running:
npm install

d. Start the frontend server:
To start the React app, run:
npm start
The React app will be available at http://localhost:3000/ in your web browser.

2. Backend Setup (Flask API)
The backend is a Flask server that handles the model prediction and serves the website.

a. Navigate to the backend directory:
In the terminal (or PowerShell), navigate to the backend directory:
cd 'Sentiment Analysis of ARABIC Tweets\backend'

b. Allow script execution (only required once on Windows):
Run this command to allow scripts to execute:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

c. Install backend dependencies:
Make sure you have Python 3.x installed. Install the required Python packages:
pip install Flask==2.0.1 Flask-Cors==3.1.1 transformers==4.9.2 pandas==1.3.2 torch==1.9.0 scikit-learn==0.24.2 requests==2.25.1

d. Start the backend server:
To start the Flask API, run:
python app.py
The Flask API will run on http://127.0.0.1:5000/ by default.

3. Running the Website
Once both the frontend and backend servers are running:

Frontend: Open your browser and go to http://localhost:3000/.

Backend: The backend Flask API should be running on http://127.0.0.1:5000/.

4. Using the Website
Model Selection: Choose a model from the dropdown (marbertv2_run, asafaya_run, arabertv2_run).

Prediction: Input a tweet in the text area and click "Predict Tweet" to get the sentiment classification.

Bulk Prediction: You can upload a CSV file to analyze multiple tweets at once by clicking "Predict CSV".

Class Distribution & Confusion Matrix: Visualize the distribution of classes and the confusion matrix for the model predictions.

5. Stopping the Servers
To stop the servers, press Ctrl + C in both the frontend and backend terminal windows.
