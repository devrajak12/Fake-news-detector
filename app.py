# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from utils import clean_text

# --- Initialize Flask app ---
app = Flask(__name__)

# --- Load model and vectorizer ONCE when server starts ---
# Loading inside the route function would reload on every request — very slow
print("Loading model...")
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load SVD only if you used PCA
# try:
#     with open('model/svd.pkl', 'rb') as f:
#         svd = pickle.load(f)
#     USE_SVD = True
# except FileNotFoundError:
#     svd = None
#     USE_SVD = False

# print("Model loaded successfully!")


def predict_news(text):
    """
    Full prediction pipeline:
    raw text → clean → vectorize → (optional SVD) → predict
    Returns label and confidence score.
    """
    # Step 1: Clean the input text
    cleaned = clean_text(text)

    # Step 2: Vectorize using trained TF-IDF
    vec = vectorizer.transform([cleaned])

    # # Step 3: Apply PCA if it was used during training
    # if USE_SVD:
    #     vec = svd.transform(vec)

    # Step 4: Predict
    prediction = model.predict(vec)[0]

    # Step 5: Get confidence score (probability)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vec)[0]
        confidence = round(float(max(proba)) * 100, 1)
    else:
        # PassiveAggressiveClassifier has no predict_proba
        decision = model.decision_function(vec)[0]
        confidence = round(min(abs(float(decision)) * 20, 99.9), 1)

    label = 'FAKE' if prediction == 1 else 'REAL'
    return label, confidence, cleaned


# --- Route 1: Home page ---
@app.route('/')
def home():
    return render_template('index.html')


# --- Route 2: Prediction from HTML form ---
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('news_text', '').strip()

    if not text:
        return render_template('index.html',
                               error="Please enter some text to analyze.")

    if len(text) < 20:
        return render_template('index.html',
                               error="Text too short. Please enter at least a sentence.",
                               news_text=text)

    label, confidence, cleaned = predict_news(text)

    return render_template('index.html',
                           prediction=label,
                           confidence=confidence,
                           news_text=text,
                           cleaned_text=cleaned)


# --- Route 3: JSON API endpoint (for developers) ---
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Accepts JSON: {"text": "your article here"}
    Returns JSON: {"label": "FAKE", "confidence": 87.3, "status": "success"}
    
    Example usage:
    curl -X POST http://localhost:5000/api/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "Scientists confirm vaccine is effective"}'
    """
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field', 'status': 'error'}), 400

    text = data['text'].strip()

    if len(text) < 20:
        return jsonify({'error': 'Text too short', 'status': 'error'}), 400

    label, confidence, cleaned = predict_news(text)

    return jsonify({
        'label':      label,
        'confidence': confidence,
        'is_fake':    label == 'FAKE',
        'status':     'success'
    })


# --- Route 4: Health check (for monitoring) ---
@app.route('/health')
def health():
    return jsonify({'status': 'running', 'model_loaded': True})


# --- Run the app ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
    # debug=True: auto-reloads when you change code
    # NEVER use debug=True in production