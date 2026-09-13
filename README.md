# 🔎 Fake Content Detection

An AI-based system for detecting fake news and social media content, combining a
trained machine learning classifier with real-time web evidence verification and
source credibility analysis — all wrapped in an interactive Streamlit app.

---

## 📌 Overview

This project analyzes a piece of news or social media text and produces:

1. A **machine learning prediction** (FAKE / REAL) using a trained text classifier
2. **Real-time web evidence** retrieved via live search, related to the claim
3. A **source credibility ranking** for each retrieved evidence source
4. A **final hybrid verdict** that combines the ML prediction with real-time
   evidence to produce a more reliable conclusion (LIKELY TRUE / LIKELY FALSE /
   UNCLEAR)
5. A **detection history dashboard** with analytics on past checks

---

## 🧠 Tech Stack

| Component               | Technology                          |
|--------------------------|--------------------------------------|
| Feature Extraction       | TF-IDF (unigrams + bigrams)          |
| Machine Learning Model   | Linear Support Vector Machine (SVM)  |
| Web Evidence Search      | Tavily Search API                    |
| Frontend / App           | Streamlit                            |
| Data Handling            | Pandas                               |
| Visualization            | Plotly                               |
| Model Persistence        | Joblib                               |

**Model Accuracy:** ~90.58% on held-out test data
**F1 Score:** ~99.62%

---

## 🚀 How It Works

1. **Input** — Paste a news article, headline, or social media claim into the app.
2. **ML Prediction** — The trained TF-IDF + Linear SVM model classifies the text
   as FAKE or REAL, along with a confidence score.
3. **Evidence Search** — The app searches the live web (via Tavily) for sources
   relevant to the claim.
4. **Credibility Ranking** — Each retrieved source is scored for credibility
   based on its domain and characteristics.
5. **Evidence Verification** — The claim is compared against the retrieved
   evidence to determine whether it is SUPPORTING, CONTRADICTING, RELEVANT, or
   has INSUFFICIENT EVIDENCE.
6. **Final Decision** — A hybrid decision engine combines the ML prediction and
   evidence status into one final verdict, with a confidence score and plain-English
   explanation.
7. **History & Analytics** — Every analysis is logged, and an analytics dashboard
   visualizes trends across all past detections.

---

## 🛠️ Project Structure

```
Fake_Content_Detection/
├── app.py                       # Main Streamlit application
├── train_fresh_model.py         # Trains the ML model from scratch
├── train_improved_model.py      # Alternate/improved training script
├── predict.py                   # Standalone CLI script to test the model directly
├── evaluate.py                  # Model evaluation script
├── advanced_evaluation.py       # Extended evaluation metrics
├── error_analysis.py            # Analyzes misclassified examples
├── prepare_improved_dataset.py  # Dataset preparation/cleaning
├── download_recent_dataset.py   # Fetches recent fake news data
├── services/
│   ├── prediction_service.py    # Loads model & makes predictions
│   ├── evidence_service.py      # Searches the web via Tavily
│   ├── claim_analysis.py        # Compares claims against evidence
│   ├── credibility_service.py   # Ranks source credibility
│   ├── final_decision_service.py# Combines ML + evidence into a verdict
│   └── history_service.py       # Saves/loads detection history (CSV)
├── data/
│   ├── True.csv                 # Original real-news training data
│   ├── Fake.csv                 # Original fake-news training data
│   └── detection_history.csv    # Logged app usage history
├── models/
│   ├── best_model.joblib        # Trained SVM model
│   └── tfidf_vectorizer.joblib  # Trained TF-IDF vectorizer
├── requirements.txt
└── .env                         # API keys (not committed to GitHub)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Fake_Content_Detection.git
cd Fake_Content_Detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the project root:

```
TAVILY_API_KEY=your_tavily_api_key_here
```

Get a free Tavily API key at [tavily.com](https://tavily.com).

### 5. Train the model (if not already trained)

```bash
python train_fresh_model.py
```

This generates `models/best_model.joblib` and `models/tfidf_vectorizer.joblib`.

### 6. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🧪 Testing the Model Independently

You can test the trained ML model on its own, without the full app, using:

```bash
python predict.py
```

This lets you type in text directly and see the model's raw FAKE/REAL prediction.

---

## ⚠️ Known Limitations

- **Dataset style bias:** The model was trained on a dataset where genuine news
  articles consistently follow a wire-service style format (e.g. starting with a
  city name and source tag, such as `WASHINGTON (Reuters) -`). As a result, the
  model can weight this structural pattern more heavily than the actual content
  when making predictions. Short headlines or real news written in a different
  style may occasionally be misclassified.
- **Short text:** The model performs best on full paragraphs of text rather than
  single-line headlines, since it was trained on full articles.
- **Evidence-based verification helps offset this:** the real-time evidence
  search and credibility layer are designed to catch and correct cases where the
  ML prediction alone may be misleading, producing a more balanced final verdict.

This is a known and documented limitation typical of TF-IDF-based text
classifiers trained on a single dataset, and is an area for future improvement
(e.g. training on a more stylistically diverse dataset).

---

## 📊 Features Summary

- ✅ Machine learning-based fake content classification
- ✅ Real-time web evidence retrieval
- ✅ Source credibility scoring
- ✅ Hybrid ML + evidence final verdict
- ✅ Detection history logging
- ✅ Interactive analytics dashboard (prediction distribution, verdict summary,
  confidence vs. credibility charts)

---

## 📄 License

This project is for educational purposes.

---

## 🙋 Author

Built as a personal/academic project exploring machine learning-based fake news
detection combined with real-time evidence verification.
