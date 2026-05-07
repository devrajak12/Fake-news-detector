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

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place `Fake.csv` and `True.csv` inside a `data/` folder.

### 4. Train the model
```bash
python train_model.py
```
This generates `model/vectorizer.pkl` and `model/model.pkl`

### 5. Run the Flask app
```bash
python app.py
```

Open browser at `http://localhost:5000`

## How It Works
1. Raw text is cleaned — URLs, HTML, punctuation removed
2. Text is tokenized and lemmatized using NLTK
3. TF-IDF vectorizer converts text to numerical features
4. Logistic Regression classifier predicts Real or Fake
5. Result shown with confidence score

## Model Performance
| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~98%   |
| F1 Score  | ~0.98  |
| Precision | ~0.98  |
| Recall    | ~0.98  |
