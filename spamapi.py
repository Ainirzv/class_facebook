from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Default labels for your forum posts
DEFAULT_LABELS = ["lost", "found", "academic", "spam", "events", "cabin no"]

@app.route("/classify", methods=["POST"])
def classify_text():
    data = request.get_json()
    text = data.get("text", "")
    labels = data.get("labels", DEFAULT_LABELS)  # Use default if not provided

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = classifier(text, candidate_labels=labels)
    
    # Extract the highest scoring label
    top_label = result['labels'][0]
    
    return jsonify({"label": top_label})

if __name__ == "__main__":
    app.run(debug=True)
