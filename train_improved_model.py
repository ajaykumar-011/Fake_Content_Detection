import re
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = MODELS_DIR / "fresh_model_metrics.txt"


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text).lower()

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


# =========================================================
# DETECT LABEL FROM FILE NAME
# =========================================================

def detect_label_from_filename(filename):

    filename = filename.lower()

    if "fake" in filename:
        return 0

    if "false" in filename:
        return 0

    if "real" in filename:
        return 1

    if "true" in filename:
        return 1

    return None


# =========================================================
# NORMALIZE LABEL
# =========================================================

def normalize_label(value):

    value = str(value).strip().lower()

    fake_values = [
        "0",
        "fake",
        "false",
        "f",
        "fabricated"
    ]

    real_values = [
        "1",
        "real",
        "true",
        "t",
        "genuine"
    ]

    if value in fake_values:
        return 0

    if value in real_values:
        return 1

    return None


# =========================================================
# FIND TEXT COLUMN
# =========================================================

def find_text_column(dataframe):

    possible_columns = [
        "text",
        "content",
        "article",
        "news",
        "body",
        "statement",
        "claim"
    ]

    columns_lower = {
        column.lower(): column
        for column in dataframe.columns
    }

    for column in possible_columns:

        if column in columns_lower:
            return columns_lower[column]

    return None


# =========================================================
# FIND TITLE COLUMN
# =========================================================

def find_title_column(dataframe):

    possible_columns = [
        "title",
        "headline",
        "subject"
    ]

    columns_lower = {
        column.lower(): column
        for column in dataframe.columns
    }

    for column in possible_columns:

        if column in columns_lower:
            return columns_lower[column]

    return None


# =========================================================
# FIND LABEL COLUMN
# =========================================================

def find_label_column(dataframe):

    possible_columns = [
        "label",
        "class",
        "target",
        "category",
        "verdict",
        "is_fake"
    ]

    columns_lower = {
        column.lower(): column
        for column in dataframe.columns
    }

    for column in possible_columns:

        if column in columns_lower:
            return columns_lower[column]

    return None


# =========================================================
# GET TRAINING FILES
# =========================================================

def get_training_files():

    files = []

    # Files directly inside data folder

    allowed_data_files = [

        "fake.csv",
        "true.csv",
        "train.csv",
        "test.csv"

    ]

    for file_path in DATA_DIR.glob("*.csv"):

        if file_path.name.lower() in allowed_data_files:

            files.append(file_path)


    # Files inside data/raw

    for file_path in RAW_DATA_DIR.glob("*.csv"):

        if (
            "history" not in file_path.name.lower()
            and "combined" not in file_path.name.lower()
            and "cleaned" not in file_path.name.lower()
        ):

            files.append(file_path)


    return files


# =========================================================
# LOAD DATASETS
# =========================================================

def load_datasets():

    print("\n" + "=" * 70)

    print("LOADING TRAINING DATASETS")

    print("=" * 70)


    csv_files = get_training_files()


    if not csv_files:

        raise FileNotFoundError(
            "No training CSV files found."
        )


    print("\nTraining files detected:")

    for file in csv_files:

        print(f" - {file}")


    all_data = []


    for csv_file in csv_files:

        print("\n" + "-" * 70)

        print(f"Reading: {csv_file.name}")


        try:

            dataframe = pd.read_csv(
                csv_file,
                low_memory=False
            )

        except Exception as error:

            print(
                f"Could not read {csv_file.name}: {error}"
            )

            continue


        print(
            f"Rows found: {len(dataframe)}"
        )

        print(
            f"Columns: {list(dataframe.columns)}"
        )


        text_column = find_text_column(
            dataframe
        )

        title_column = find_title_column(
            dataframe
        )

        label_column = find_label_column(
            dataframe
        )

        filename_label = detect_label_from_filename(
            csv_file.name
        )


        # =============================================
        # CREATE TEXT
        # =============================================

        if text_column:

            text_data = dataframe[
                text_column
            ].fillna("").astype(str)


            if title_column:

                text_data = (

                    dataframe[
                        title_column
                    ].fillna("").astype(str)

                    + " "

                    + text_data

                )


        elif title_column:

            text_data = dataframe[
                title_column
            ].fillna("").astype(str)


        else:

            print(
                "Skipping: No usable text column."
            )

            continue


        # =============================================
        # CREATE LABEL
        # =============================================

        if label_column:

            labels = dataframe[
                label_column
            ].apply(
                normalize_label
            )


        elif filename_label is not None:

            labels = pd.Series(
                [filename_label] * len(dataframe)
            )


        else:

            print(
                "Skipping: Could not determine labels."
            )

            continue


        processed = pd.DataFrame({

            "text": text_data,

            "label": labels

        })


        processed = processed.dropna()


        print(
            f"Usable rows: {len(processed)}"
        )


        all_data.append(
            processed
        )


    if not all_data:

        raise ValueError(
            "No usable datasets were loaded."
        )


    combined_data = pd.concat(

        all_data,

        ignore_index=True

    )


    return combined_data


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model():

    MODELS_DIR.mkdir(
        exist_ok=True
    )


    # =====================================================
    # LOAD DATA
    # =====================================================

    data = load_datasets()


    print("\n" + "=" * 70)

    print("DATA PREPROCESSING")

    print("=" * 70)


    print(
        f"\nTotal rows loaded: {len(data)}"
    )


    # Remove empty text

    data["text"] = (

        data["text"]

        .astype(str)

        .str.strip()

    )


    data = data[
        data["text"].str.len() > 10
    ]


    # Clean text

    print("\nCleaning text...")


    data["clean_text"] = (

        data["text"]

        .apply(clean_text)

    )


    # Remove duplicates

    before_duplicates = len(data)


    data = data.drop_duplicates(
        subset=["clean_text"]
    )


    duplicates_removed = (
        before_duplicates - len(data)
    )


    print(
        f"Duplicates removed: {duplicates_removed}"
    )


    # Final cleaning

    data = data[
        data["clean_text"].str.len() > 10
    ]


    data = data.dropna(
        subset=["label"]
    )


    data["label"] = (
        data["label"].astype(int)
    )


    print(
        f"Final dataset size: {len(data)}"
    )


    # =====================================================
    # CLASS DISTRIBUTION
    # =====================================================

    fake_count = (
        data["label"] == 0
    ).sum()


    real_count = (
        data["label"] == 1
    ).sum()


    print("\nDATASET DISTRIBUTION")

    print(
        f"FAKE: {fake_count}"
    )

    print(
        f"REAL: {real_count}"
    )


    if fake_count == 0 or real_count == 0:

        raise ValueError(
            "Dataset must contain BOTH FAKE and REAL samples."
        )


    # =====================================================
    # FEATURES AND LABELS
    # =====================================================

    X = data["clean_text"]

    y = data["label"]


    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================

    print("\n" + "=" * 70)

    print("CREATING TRAIN / TEST SPLIT")

    print("=" * 70)


    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42,

            stratify=y

        )

    )


    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )


    # =====================================================
    # TF-IDF
    # =====================================================

    print("\n" + "=" * 70)

    print("CREATING TF-IDF FEATURES")

    print("=" * 70)


    vectorizer = TfidfVectorizer(

        max_features=100000,

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95,

        sublinear_tf=True,

        stop_words="english"

    )


    X_train_features = (

        vectorizer.fit_transform(
            X_train
        )

    )


    X_test_features = (

        vectorizer.transform(
            X_test
        )

    )


    print(
        f"Total features: "
        f"{len(vectorizer.get_feature_names_out())}"
    )


    # =====================================================
    # LINEAR SVM
    # =====================================================

    print("\n" + "=" * 70)

    print("TRAINING LINEAR SVM")

    print("=" * 70)


    model = LinearSVC(

        C=1.0,

        class_weight="balanced",

        random_state=42

    )


    model.fit(

        X_train_features,

        y_train

    )


    # =====================================================
    # EVALUATION
    # =====================================================

    predictions = model.predict(
        X_test_features
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    f1 = f1_score(

        y_test,

        predictions,

        average="weighted"

    )


    report = classification_report(

        y_test,

        predictions,

        target_names=[
            "FAKE",
            "REAL"
        ]

    )


    matrix = confusion_matrix(

        y_test,

        predictions

    )


    print("\n" + "=" * 70)

    print("MODEL RESULTS")

    print("=" * 70)


    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"F1 Score: {f1 * 100:.2f}%"
    )


    print(
        "\nClassification Report:\n"
    )

    print(report)


    print(
        "\nConfusion Matrix:"
    )

    print(matrix)


    # =====================================================
    # SAVE MODEL
    # =====================================================

    print("\n" + "=" * 70)

    print("SAVING MODEL")

    print("=" * 70)


    joblib.dump(
        model,
        MODEL_PATH
    )


    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )


    # =====================================================
    # SAVE METRICS
    # =====================================================

    metrics_content = f"""
FRESH MODEL EVALUATION REPORT
=============================

Dataset Size: {len(data)}

Fake Samples: {fake_count}

Real Samples: {real_count}

Training Samples: {len(X_train)}

Testing Samples: {len(X_test)}

Accuracy: {accuracy * 100:.2f}%

F1 Score: {f1 * 100:.2f}%

Confusion Matrix:

{matrix}

Classification Report:

{report}
"""


    with open(

        METRICS_PATH,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(
            metrics_content
        )


    print("\n" + "=" * 70)

    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")

    print("=" * 70)


    print(
        f"\nModel:\n{MODEL_PATH}"
    )

    print(
        f"\nVectorizer:\n{VECTORIZER_PATH}"
    )

    print(
        f"\nMetrics:\n{METRICS_PATH}"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    train_model()