"""
Flask app that serves a live next-word predictor UI.

Before running this, train and save the model once:
    python train_model.py

Then start the server:
    python app.py
"""

import json
import os
import pickle

import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ---------------------------------------------------------------------------
# Load the trained model, tokenizer, and config once at startup
# ---------------------------------------------------------------------------
model = load_model(os.path.join(MODEL_DIR, "next_word_model.keras"))

with open(os.path.join(MODEL_DIR, "tokenizer.pickle"), "rb") as f:
    tokenizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    config = json.load(f)

max_seq_len = config["max_seq_len"]

# Reverse lookup: index -> word
index_to_word = {index: word for word, index in tokenizer.word_index.items()}


def predict_top_words(seed_text, top_k=3):
    """Return the top_k most likely next words with their probabilities."""
    if not seed_text.strip():
        return []

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    if not token_list:
        return []

    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding="pre")
    predicted_probs = model.predict(token_list, verbose=0)[0]

    top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        word = index_to_word.get(idx)
        if word:
            results.append({"word": word, "confidence": float(predicted_probs[idx])})
    return results


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    seed_text = data.get("text", "")
    predictions = predict_top_words(seed_text, top_k=3)
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(debug=True, port=5000)