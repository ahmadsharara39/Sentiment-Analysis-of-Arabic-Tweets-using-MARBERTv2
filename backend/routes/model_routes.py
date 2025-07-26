# backend/routes/model_routes.py

from flask import Blueprint, request, jsonify
import os

from utils.predict_utils import predict_from_csv, predict_single_tweet
from utils.evaluate_utils import get_metrics

model_bp = Blueprint('model_bp', __name__)
MODEL_DIR = "outputs"


@model_bp.route('/models', methods=['GET'])
def list_models():
    """List all saved model directories plus 'ensemble'."""
    models = [d for d in os.listdir(MODEL_DIR)
              if os.path.isdir(os.path.join(MODEL_DIR, d))]
    models.append("ensemble")
    return jsonify(models)


@model_bp.route('/predict', methods=['POST'])
def predict_bulk():
    """
    Bulk CSV prediction.
    Expects form-data:
      - 'model': the model directory name
      - 'file':   the uploaded CSV (must have a text column)
    Returns JSON:
      {
        "predictions": [
          { "text": "...", "label": 1, "confidence": 0.92 }, …
        ],
        "metrics": {
          "accuracy": 0.85,
          "f1_score": 0.84,
          "precision": 0.86,
          "recall": 0.85,
          "confusion_matrix": [[10,1,0],[2,15,1],[0,1,9]]
        }
      }
      or if no label column is present:
      {
        "predictions": [ … ],
        "metrics": null
      }
    """
    model_name = request.form.get('model', '').strip()
    file_obj   = request.files.get('file')
    if not model_name or not file_obj:
        return jsonify({"error": "Missing model name or file upload"}), 400

    # Save upload to uploads/ folder
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file_obj.filename)
    file_obj.save(filepath)

    # 1) Run predictions and get cleaned DataFrame
    try:
        results, cleaned_df = predict_from_csv(filepath, model_name)
    except Exception as err:
        print("❌ predict_utils error:", err)
        return jsonify({"error": str(err)}), 500

    # 2) Compute metrics (with confusion matrix)
    try:
        metrics = get_metrics(results, cleaned_df)
    except Exception as err:
        # If CSV lacks a label column, skip metrics instead of erroring
        print("⚠️ Skipping metrics:", err)
        metrics = None

    return jsonify({
        "predictions": results,
        "metrics":     metrics
    })


@model_bp.route('/predict-single', methods=['POST'])
def predict_single():
    """
    Single-tweet prediction endpoint.
    Expects JSON body: { "tweet": "...", "model": "<model_name>" }
    Returns JSON: { "label": int, "confidence": float }
    """
    payload    = request.get_json(force=True) or {}
    tweet      = payload.get('tweet', '').strip()
    model_name = payload.get('model', '').strip()

    if not tweet or not model_name:
        return jsonify({"error": "Missing 'tweet' text or 'model' name"}), 400

    try:
        label, confidence = predict_single_tweet(tweet, model_name)
        return jsonify({"label": label, "confidence": confidence})
    except Exception as err:
        print("❌ predict-single error:", err)
        return jsonify({"error": str(err)}), 500
