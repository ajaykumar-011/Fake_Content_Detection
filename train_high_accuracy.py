import re
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"

TRUE_PATH = DATA_DIR / "True.csv"
FAKE_PATH = DATA_DIR / "Fake.csv"
RECENT_FAKE_PATH = RAW_DIR / "recent_fake_news.csv"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"

MODELS_DIR.mkdir(exist_ok=True)


# ==================================================
# TEXT CLEANING
# ==================================================

def clean_text(text):

    text = str(text).lower()

    # Remove wire-service names (source-identity shortcut)

    text = re.sub(r"\breuters\b", " ", text)
    text = re.sub(r"\b(ap|afp|bloomberg)\b", " ", text)

    # Remove common wire-service boilerplate/bylines -
    # this is structural noise, not content about
    # truthfulness, so removing it is fair cleanup.

    text = re.sub(r"\(?\s*reporting by.*?(\)|\.|$)", " ", text)
    text = re.sub(r"\(?\s*editing by.*?(\)|\.|$)", " ", text)
    text = re.sub(r"\(?\s*additional reporting by.*?(\)|\.|$)", " ", text)
    text = re.sub(r"featured image via.*", " ", text)
    text = re.sub(r"image via.*", " ", text)

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==================================================
# CREATE TEXT COLUMN
# ==================================================

def create_text_column(df):

    possible_columns = ["text", "content", "article", "news", "body", "title"]
    available_columns = [c for c in possible_columns if c in df.columns]

    if not available_columns:
        raise ValueError(f"No text column found. Available columns: {list(df.columns)}")

    combined_text = None

    for column in available_columns:
        if combined_text is None:
            combined_text = df[column].fillna("").astype(str)
        else:
            combined_text = combined_text + " " + df[column].fillna("").astype(str)

    df["content"] = combined_text
    return df


# ==================================================
# LOAD DATASETS
# ==================================================

print("\nLoading TRUE dataset...")
true_df = pd.read_csv(TRUE_PATH)
true_df = create_text_column(true_df)
true_df["label"] = 1
print(f"TRUE news loaded: {len(true_df)}")

print("\nLoading FAKE dataset...")
fake_df = pd.read_csv(FAKE_PATH)
fake_df = create_text_column(fake_df)
fake_df["label"] = 0
print(f"FAKE news loaded: {len(fake_df)}")

recent_fake_df = None

if RECENT_FAKE_PATH.exists():
    print("\nLoading recent fake news dataset...")
    recent_fake_df = pd.read_csv(RECENT_FAKE_PATH)
    recent_fake_df = create_text_column(recent_fake_df)
    recent_fake_df["label"] = 0
    print(f"Recent fake news loaded: {len(recent_fake_df)}")
else:
    print("\nWARNING: recent_fake_news.csv not found.")


# ==================================================
# COMBINE DATASETS
# ==================================================

true_data = true_df[["content", "label"]].copy()
fake_data = fake_df[["content", "label"]].copy()
datasets = [true_data, fake_data]

if recent_fake_df is not None:
    recent_fake_data = recent_fake_df[["content", "label"]].copy()
    datasets.append(recent_fake_data)

print("\nCombining datasets...")
df = pd.concat(datasets, ignore_index=True)


# ==================================================
# CLEAN DATA
# ==================================================

print("Cleaning text...")

df["content"] = df["content"].fillna("").astype(str).apply(clean_text)
df = df[df["content"].str.len() > 20]
df = df.drop_duplicates(subset=["content"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal dataset size: {len(df)}")
print("\nClass distribution:")
print(df["label"].value_counts())


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# ==================================================
# COMBINED WORD + CHARACTER TF-IDF + FEATURE SELECTION
# ==================================================

print("\nBuilding combined word + character TF-IDF features...")

word_vectorizer = TfidfVectorizer(
    max_features=150000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

char_vectorizer = TfidfVectorizer(
    max_features=50000,
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

combined_features = FeatureUnion([
    ("word", word_vectorizer),
    ("char", char_vectorizer)
])

print("Fitting feature extraction...")
X_train_raw = combined_features.fit_transform(X_train)
X_test_raw = combined_features.transform(X_test)

print(f"Raw combined feature matrix shape: {X_train_raw.shape}")

print("\nSelecting most informative features (chi-squared)...")

selector = SelectKBest(chi2, k=min(80000, X_train_raw.shape[1]))

X_train_selected = selector.fit_transform(X_train_raw, y_train)
X_test_selected = selector.transform(X_test_raw)

print(f"Selected feature matrix shape: {X_train_selected.shape}")

# Bundle feature extraction + selection into one object so
# prediction_service.py can keep calling vectorizer.transform()
# exactly as before, with no other code changes needed.

vectorizer = Pipeline([
    ("features", combined_features),
    ("select", selector)
])


# ==================================================
# HYPERPARAMETER TUNING - LINEAR SVM
# ==================================================

print("\nTuning Linear SVM on selected features (this will take a few minutes)...")

param_grid = {
    "C": [0.1, 0.3, 0.5, 1, 2, 5, 10]
}

grid = GridSearchCV(
    LinearSVC(class_weight="balanced", max_iter=5000),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=1
)

grid.fit(X_train_selected, y_train)

best_model = grid.best_estimator_

print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_ * 100:.2f}%")


# ==================================================
# FINAL EVALUATION
# ==================================================

print("\nEvaluating on test set...")

predictions = best_model.predict(X_test_selected)
accuracy = accuracy_score(y_test, predictions)

print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, predictions, target_names=["FAKE", "REAL"]))


# ==================================================
# SAVE MODEL
# ==================================================

print("\nSaving model...")

joblib.dump(best_model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("\n====================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("====================================")
print(f"\nModel saved to:\n{MODEL_PATH}")
print(f"\nVectorizer saved to:\n{VECTORIZER_PATH}")