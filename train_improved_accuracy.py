import re
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
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

    # Remove wire-service / dateline artifacts so the
    # model can't shortcut on them (this was the source
    # of the earlier accuracy inflation / bias).

    text = re.sub(r"\breuters\b", " ", text)
    text = re.sub(r"\b(ap|afp|bloomberg)\b", " ", text)

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==================================================
# CREATE TEXT COLUMN
# ==================================================

def create_text_column(df):

    possible_columns = [
        "text", "content", "article", "news", "body", "title"
    ]

    available_columns = [
        column for column in possible_columns if column in df.columns
    ]

    if not available_columns:
        raise ValueError(
            f"No text column found. Available columns: {list(df.columns)}"
        )

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

df["content"] = (
    df["content"]
    .fillna("")
    .astype(str)
    .apply(clean_text)
)

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
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==================================================
# TF-IDF VECTORIZATION (richer features)
# ==================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=200000,
    stop_words="english",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# ==================================================
# HYPERPARAMETER TUNING - LINEAR SVM
# ==================================================

print("\nTuning Linear SVM (this may take a few minutes)...")

svm_param_grid = {
    "C": [0.1, 0.5, 1, 2, 5, 10]
}

svm_grid = GridSearchCV(
    LinearSVC(class_weight="balanced", max_iter=5000),
    svm_param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

svm_grid.fit(X_train_vectorized, y_train)

best_svm = svm_grid.best_estimator_

print(f"Best SVM params: {svm_grid.best_params_}")
print(f"Best SVM CV accuracy: {svm_grid.best_score_ * 100:.2f}%")


# ==================================================
# COMPARE WITH LOGISTIC REGRESSION
# ==================================================

print("\nTraining Logistic Regression for comparison...")

logreg = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    C=1.0
)

logreg.fit(X_train_vectorized, y_train)

logreg_preds = logreg.predict(X_test_vectorized)
logreg_acc = accuracy_score(y_test, logreg_preds)

print(f"Logistic Regression Test Accuracy: {logreg_acc * 100:.2f}%")


# ==================================================
# COMPARE WITH NAIVE BAYES
# ==================================================

print("\nTraining Complement Naive Bayes for comparison...")

nb = ComplementNB()
nb.fit(X_train_vectorized, y_train)

nb_preds = nb.predict(X_test_vectorized)
nb_acc = accuracy_score(y_test, nb_preds)

print(f"Naive Bayes Test Accuracy: {nb_acc * 100:.2f}%")


# ==================================================
# FINAL EVALUATION - BEST SVM
# ==================================================

print("\nEvaluating tuned SVM on test set...")

svm_preds = best_svm.predict(X_test_vectorized)
svm_acc = accuracy_score(y_test, svm_preds)

print(f"Tuned SVM Test Accuracy: {svm_acc * 100:.2f}%")


# ==================================================
# PICK THE BEST MODEL
# ==================================================

results = {
    "SVM": (best_svm, svm_acc),
    "LogisticRegression": (logreg, logreg_acc),
    "NaiveBayes": (nb, nb_acc)
}

best_model_name = max(results, key=lambda name: results[name][1])
best_model, best_accuracy = results[best_model_name]

print(f"\n{'=' * 50}")
print(f"BEST MODEL: {best_model_name}")
print(f"BEST ACCURACY: {best_accuracy * 100:.2f}%")
print(f"{'=' * 50}")

final_preds = best_model.predict(X_test_vectorized)

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    final_preds,
    target_names=["FAKE", "REAL"]
))


# ==================================================
# SAVE BEST MODEL
# ==================================================

print("\nSaving model...")

joblib.dump(best_model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("\n====================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("====================================")
print(f"\nModel saved to:\n{MODEL_PATH}")
print(f"\nVectorizer saved to:\n{VECTORIZER_PATH}")