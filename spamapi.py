from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# Initialize the distilBERT model for zero-shot classification
classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased")

# Default labels for your forum posts
DEFAULT_LABELS = ["lost", "found", "academic", "spam", "events", "cabin no"]

@app.route("/classify", methods=["POST"])
def classify_text():
    data = request.get_json()
    text = data.get("text", "")
    labels = data.get("labels", DEFAULT_LABELS)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = classifier(text, candidate_labels=labels)
    top_label = result['labels'][0]
    
    return jsonify({"label": top_label})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

