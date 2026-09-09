import re
from pathlib import Path

import joblib


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_model.joblib"
)


VECTORIZER_PATH = (
    BASE_DIR
    / "models"
    / "tfidf_vectorizer.joblib"
)


# ==========================================================
# TEXT CLEANING
# ==========================================================

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


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_prediction_model():

    model = joblib.load(
        MODEL_PATH
    )


    vectorizer = joblib.load(
        VECTORIZER_PATH
    )


    return model, vectorizer


# ==========================================================
# PREDICT CONTENT
# ==========================================================

def predict_content(text):

    model, vectorizer = (
        load_prediction_model()
    )


    cleaned_text = clean_text(
        text
    )


    features = vectorizer.transform(
        [cleaned_text]
    )


    prediction = model.predict(
        features
    )[0]


    if int(prediction) == 0:

        result = "FAKE"

    else:

        result = "REAL"


    # Default values

    decision_score = None

    confidence = 50.0

    prediction_strength = "LOW"


    # ======================================================
    # LINEAR SVM CONFIDENCE
    # ======================================================

    if hasattr(
        model,
        "decision_function"
    ):

        decision_score = float(

            model.decision_function(
                features
            )[0]

        )


        absolute_score = abs(
            decision_score
        )


        # Approximate confidence

        confidence = min(

            round(
                50 + (
                    absolute_score * 20
                ),
                2
            ),

            99.0

        )


        if absolute_score >= 1.5:

            prediction_strength = "HIGH"

        elif absolute_score >= 0.6:

            prediction_strength = "MODERATE"

        else:

            prediction_strength = "LOW"


    # ======================================================
    # PROBABILITY BASED MODELS
    # ======================================================

    elif hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                features
            )[0]
        )


        confidence = round(

            max(probabilities)
            * 100,

            2

        )


        if confidence >= 80:

            prediction_strength = "HIGH"

        elif confidence >= 60:

            prediction_strength = "MODERATE"

        else:

            prediction_strength = "LOW"


    return {

        "prediction": result,

        "decision_score": decision_score,

        "confidence": confidence,

        "prediction_strength": prediction_strength

    }