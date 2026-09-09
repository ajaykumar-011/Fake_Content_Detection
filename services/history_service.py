import csv
import os
from datetime import datetime


# ==================================================
# HISTORY FILE LOCATION
# ==================================================

HISTORY_FILE = "data/detection_history.csv"


# ==================================================
# CREATE HISTORY FILE
# ==================================================

def initialize_history():

    os.makedirs(
        "data",
        exist_ok=True
    )


    if not os.path.exists(HISTORY_FILE):

        with open(
            HISTORY_FILE,
            mode="w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)


            writer.writerow([
                "timestamp",
                "content",
                "ml_prediction",
                "evidence_status",
                "evidence_score",
                "source_credibility",
                "final_verdict",
                "final_confidence"
            ])


# ==================================================
# SAVE DETECTION HISTORY
# ==================================================

def save_detection_history(
    content,
    prediction,
    evidence_status,
    evidence_score,
    source_credibility,
    final_verdict,
    final_confidence
):

    initialize_history()


    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    with open(
        HISTORY_FILE,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)


        writer.writerow([
            timestamp,
            content,
            prediction,
            evidence_status,
            evidence_score,
            source_credibility,
            final_verdict,
            final_confidence
        ])


# ==================================================
# LOAD HISTORY
# ==================================================

def load_detection_history():

    initialize_history()


    history = []


    with open(
        HISTORY_FILE,
        mode="r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)


        for row in reader:

            history.append(row)


    return history