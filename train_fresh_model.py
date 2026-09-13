import re
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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
RECENT_REAL_PATH = RAW_DIR / "recent_real_news.csv"


MODEL_PATH = MODELS_DIR / "best_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"


# ==================================================
# CREATE MODELS DIRECTORY
# ==================================================

MODELS_DIR.mkdir(exist_ok=True)


# ==================================================
# TEXT CLEANING
# ==================================================

def clean_text(text):

    text = str(text).lower()

    # ==============================================
    # REMOVE WIRE-SERVICE / DATELINE ARTIFACTS
    # ==============================================
    #
    # The original dataset almost always tags real
    # articles with a wire-service name (e.g. "reuters").
    # Left in, the model learns to key off this single
    # word instead of genuine content signals. We strip
    # it out here so the model must learn from the text
    # itself, not this shortcut.

    text = re.sub(
        r"\breuters\b",
        " ",
        text
    )

    text = re.sub(
        r"\b(ap|afp|bloomberg)\b",
        " ",
        text
    )

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==================================================
# CREATE TEXT COLUMN
# ==================================================

def create_text_column(df):

    possible_columns = [
        "text",
        "content",
        "article",
        "news",
        "body",
        "title"
    ]

    available_columns = [
        column
        for column in possible_columns
        if column in df.columns
    ]

    if not available_columns:

        raise ValueError(
            f"No text column found. "
            f"Available columns: {list(df.columns)}"
        )


    combined_text = None

    for column in available_columns:

        if combined_text is None:

            combined_text = (
                df[column]
                .fillna("")
                .astype(str)
            )

        else:

            combined_text = (
                combined_text
                + " "
                + df[column]
                .fillna("")
                .astype(str)
            )


    df["content"] = combined_text

    return df


# ==================================================
# LOAD ORIGINAL TRUE NEWS
# ==================================================

print("\nLoading TRUE dataset...")

true_df = pd.read_csv(
    TRUE_PATH
)

true_df = create_text_column(
    true_df
)

true_df["label"] = 1


print(
    f"TRUE news loaded: {len(true_df)}"
)


# ==================================================
# LOAD ORIGINAL FAKE NEWS
# ==================================================

print("\nLoading FAKE dataset...")

fake_df = pd.read_csv(
    FAKE_PATH
)

fake_df = create_text_column(
    fake_df
)

fake_df["label"] = 0


print(
    f"FAKE news loaded: {len(fake_df)}"
)


# ==================================================
# LOAD RECENT FAKE NEWS DATASET
# ==================================================

recent_fake_df = None

if RECENT_FAKE_PATH.exists():

    print(
        "\nLoading recent fake news dataset..."
    )

    recent_fake_df = pd.read_csv(
        RECENT_FAKE_PATH
    )

    print(
        "Recent dataset columns:"
    )

    print(
        list(recent_fake_df.columns)
    )


    recent_fake_df = create_text_column(
        recent_fake_df
    )

    recent_fake_df["label"] = 0


    print(
        f"Recent fake news loaded: "
        f"{len(recent_fake_df)}"
    )

else:

    print(
        "\nWARNING: recent_fake_news.csv "
        "not found."
    )


# ==================================================
# LOAD RECENT REAL NEWS DATASET
# ==================================================

recent_real_df = None

if RECENT_REAL_PATH.exists():

    print(
        "\nLoading recent real news dataset..."
    )

    recent_real_df = pd.read_csv(
        RECENT_REAL_PATH
    )

    print(
        "Recent real dataset columns:"
    )

    print(
        list(recent_real_df.columns)
    )


    recent_real_df = create_text_column(
        recent_real_df
    )

    recent_real_df["label"] = 1


    print(
        f"Recent real news loaded: "
        f"{len(recent_real_df)}"
    )

else:

    print(
        "\nWARNING: recent_real_news.csv not found. "
        "Run download_recent_real_news.py first to "
        "generate it - without it, the model will only "
        "see old (2017-2018) examples of real news."
    )


# ==================================================
# SELECT REQUIRED COLUMNS
# ==================================================

true_data = true_df[
    ["content", "label"]
].copy()


fake_data = fake_df[
    ["content", "label"]
].copy()


datasets = [
    true_data,
    fake_data
]


if recent_fake_df is not None:

    recent_fake_data = recent_fake_df[
        ["content", "label"]
    ].copy()

    datasets.append(
        recent_fake_data
    )


if recent_real_df is not None:

    recent_real_data = recent_real_df[
        ["content", "label"]
    ].copy()

    datasets.append(
        recent_real_data
    )


# ==================================================
# COMBINE DATASETS
# ==================================================

print(
    "\nCombining datasets..."
)


df = pd.concat(
    datasets,
    ignore_index=True
)


# ==================================================
# CLEAN DATA
# ==================================================

print(
    "Cleaning text..."
)


df["content"] = (
    df["content"]
    .fillna("")
    .astype(str)
    .apply(clean_text)
)


df = df[
    df["content"].str.len() > 20
]


# Remove duplicate articles

df = df.drop_duplicates(
    subset=["content"]
)


df = df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


print(
    f"\nFinal dataset size: {len(df)}"
)


print(
    "\nClass distribution:"
)


print(
    df["label"]
    .value_counts()
)


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X = df["content"]

y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


# ==================================================
# TF-IDF VECTORIZATION
# ==================================================

print(
    "\nCreating TF-IDF features..."
)


vectorizer = TfidfVectorizer(

    max_features=100000,

    stop_words="english",

    ngram_range=(1, 2),

    min_df=2,

    max_df=0.95,

    sublinear_tf=True

)


X_train_vectorized = vectorizer.fit_transform(
    X_train
)


X_test_vectorized = vectorizer.transform(
    X_test
)


# ==================================================
# TRAIN LINEAR SVM MODEL
# ==================================================

print(
    "\nTraining Linear SVM model..."
)


model = LinearSVC(

    C=1.0,

    class_weight="balanced"

)


model.fit(

    X_train_vectorized,

    y_train

)


# ==================================================
# MODEL EVALUATION
# ==================================================

print(
    "\nEvaluating model..."
)


predictions = model.predict(
    X_test_vectorized
)


accuracy = accuracy_score(

    y_test,

    predictions

)


print(
    f"\nModel Accuracy: "
    f"{accuracy * 100:.2f}%"
)


print(
    "\nClassification Report:\n"
)


print(

    classification_report(

        y_test,

        predictions,

        target_names=[
            "FAKE",
            "REAL"
        ]

    )

)


# ==================================================
# SAVE MODEL
# ==================================================

print(
    "\nSaving model..."
)


joblib.dump(

    model,

    MODEL_PATH

)


joblib.dump(

    vectorizer,

    VECTORIZER_PATH

)


print(
    "\n===================================="
)

print(
    "MODEL TRAINING COMPLETED SUCCESSFULLY"
)

print(
    "===================================="
)


print(
    f"\nModel saved to:\n{MODEL_PATH}"
)


print(
    f"\nVectorizer saved to:\n{VECTORIZER_PATH}"
)