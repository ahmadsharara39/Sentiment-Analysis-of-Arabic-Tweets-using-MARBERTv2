<!-- CHANGELOG.md -->

# Changelog

## 2025-06-10
- Completed the development of the website for sentiment analysis, integrating Flask backend and React frontend.
- Enhanced the user interface with modern, user-friendly design and charts, such as prediction results and confusion matrix.
- Added the "Class Distribution" and "Prediction Results" features to display model outputs in a neat format.

## 2025-06-01 – 2025-06-10
- Integrated model selector, prediction tables, and real-time metrics dashboard into the website.
- Implemented backend API calls for model selection and prediction functionalities.
- Styled the website to be modern and visually appealing with custom themes and responsive layouts.

## 2025-05-31
- Finalized research on how to implement the project using Flask and React.

## 2025-05-30
- Designed and tested API for model prediction using Flask backend and React frontend.

## 2025-05-24
- Documented every structural change in a text/DOCX file for inclusion in the final report.

## 2025-05-23
- Final testing and minor documentation refinements.

## 2025-05-22
- Selected the best-performing models and implemented the logits-averaging ensemble in `ensemble.py`.

## 2025-05-21
- Benchmarked multiple transformer architectures on the sampled dataset and recorded their metrics.

## 2025-05-20
- Updated `train.py` to accept simple console input for model selection at runtime.

## 2025-05-19
- Modified `predict.py` to call the new `data_cleaning.py` module.  
- Developed additional scripts for other Arabic transformer models (e.g. AraBERT, AraBERTv2).

## 2025-05-18
- Refactored the codebase into four modules:  
  - `data_cleaning.py`  
  - `model.py`  
  - `train.py`  
  - `evaluate.py`

## 2025-05-17
- Created a smaller sampled dataset (`Sampled_Dataset.csv`) for quicker, less resource-heavy training.

## 2025-05-15
- Updated `evaluate.py` to run only on the held-out test set (not the full dataset).

## 2025-05-14
- Reduced default training epochs from 4 → 2 in `train.py` for faster iteration.

## 2025-05-13
- Added a `--skip-training` flag in `train.py` to bypass retraining when checkpoints exist.

## 2025-05-12
- Set `Test_Size = 0.2` for a more meaningful held-out split in `train.py`.

## 2025-05-11
- Saved the tokenizer during training to ensure proper inference compatibility.

## 2025-05-10
- Authored “Phase Five” and “Conclusion” sections for integration into the final report.

## 2025-05-09
- Visualized prediction class distribution with a bar chart.

## 2025-05-08
- Enhanced `predict.py` to output `Tweet_id`, cleaned `Text`, and the predicted sentiment label.

## 2025-05-07
- Achieved **89% accuracy** and **0.87 macro F1** on the held-out evaluation set.

## 2025-05-06
- Trained the model for 4 epochs with optimizer tuning and EarlyStopping (patience=2).

## 2025-05-05
- Implemented stratified sampling for fair train/test splits.

## 2025-05-04
- Applied class-weighting in the loss function to address sentiment imbalance.

## 2025-05-03
- Upgraded the base model architecture to **UBC-NLP/MARBERTv2**.
