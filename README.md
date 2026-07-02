# Next Word Predictor (LSTM + Flask)

A simple web app that predicts the next word as you type, powered by an LSTM language model served through a Flask backend.

## Features
- LSTM-based next-word prediction trained on a custom text corpus
- Live predictions in the browser as you type (no page reload)
- Top-3 word suggestions with confidence scores
- Click a suggestion or press `Tab` to insert it

## Tech Stack
- **Model:** TensorFlow / Keras (LSTM)
- **Backend:** Flask
- **Frontend:** HTML, CSS, vanilla JS

## Setup

```bash
git clone https://github.com/its-altaf/next-word-predictor.git
cd next-word-predictor

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

python train_model.py        # trains and saves the model (run once)
python app.py                # starts the server
```

Open **http://127.0.0.1:5000** in your browser.

## Project Structure
```
next-word-app/
├── app.py              # Flask server + prediction endpoint
├── train_model.py      # Trains and saves the LSTM model
├── requirements.txt
└── templates/
    └── index.html      # UI
```

## Notes
- The included corpus in `train_model.py` is a tiny sample — swap it for your own text data and re-run `train_model.py` for meaningful predictions.

