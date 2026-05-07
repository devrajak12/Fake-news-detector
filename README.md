# Fake News Detector

A machine learning web application that classifies news articles as 
Real or Fake using NLP and Logistic Regression.

## Tech Stack
- Python 3.x
- Flask (web framework)
- scikit-learn (ML model)
- NLTK (text preprocessing)
- TF-IDF Vectorizer (feature extraction)

## Project Structure
fake_news_detector/
├── model/              ← trained model saved here (generated locally)
├── templates/
│   └── index.html
├── app.py
├── utils.py
├── train_model.py      ← run this first to generate model files
└── requirements.txt
