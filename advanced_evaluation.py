import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

X_test = joblib.load("models/X_test_tfidf.joblib")
y_test = joblib.load("models/y_test.joblib")

# --------------------------------------------------
# LOAD BEST MODEL
# --------------------------------------------------

model = joblib.load("models/best_model.joblib")

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

predictions = model.predict(X_test)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("=" * 70)
print("           STEP 14 - ADVANCED MODEL EVALUATION")
print("=" * 70)

print("\nModel: Linear SVM")

print("\nAccuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print("\n" + "-" * 70)
print("CLASSIFICATION REPORT")
print("-" * 70)

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Fake", "Real"],
        zero_division=0
    )
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, predictions)

print("\n" + "-" * 70)
print("CONFUSION MATRIX")
print("-" * 70)

print(cm)

# --------------------------------------------------
# SAVE CONFUSION MATRIX IMAGE
# --------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake", "Real"]
)

display.plot()

plt.title("Linear SVM - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------

results = pd.DataFrame({
    "Model": ["Linear SVM"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1]
})

results.to_csv(
    "models/final_evaluation.csv",
    index=False
)

# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------

print("\n" + "=" * 70)
print("STEP 14 COMPLETE")
print("=" * 70)

print("\nFiles created:")

print("models/confusion_matrix.png")
print("models/final_evaluation.csv")

print("\nModel validation completed successfully.")