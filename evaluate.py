import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

X_test = joblib.load("models/X_test_tfidf.joblib")
y_test = joblib.load("models/y_test.joblib")

models = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Linear SVM": "models/linear_svm.joblib",
    "Naive Bayes": "models/naive_bayes.joblib",
    "Random Forest": "models/random_forest.joblib"
}

results = []

print("=" * 70)
print("              STEP 11 - MODEL EVALUATION")
print("=" * 70)

for name, path in models.items():

    model = joblib.load(path)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4)
    })

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Fake", "Real"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


df = pd.DataFrame(results)

df = df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

df.to_csv(
    "models/evaluation_results.csv",
    index=False
)

print("\n" + "=" * 70)
print("STEP 11 COMPLETE")
print("=" * 70)

print("\nFINAL MODEL EVALUATION:")
print(df.to_string(index=False))

print("\nBest Model:", df.iloc[0]["Model"])
print("Best F1 Score:", df.iloc[0]["F1 Score"])

print("\nResults saved to:")
print("models/evaluation_results.csv")