import joblib
import pandas as pd

from sklearn.metrics import accuracy_score

print("=" * 70)
print("                 STEP 12 - ERROR ANALYSIS")
print("=" * 70)

# Load test data
test_data = pd.read_csv("data/test.csv")

# Load test TF-IDF features and labels
X_test = joblib.load("models/X_test_tfidf.joblib")
y_test = joblib.load("models/y_test.joblib")

# Load best model
model = joblib.load("models/linear_svm.joblib")

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nBest Model: Linear SVM")
print("Test Samples:", len(y_test))
print("Accuracy:", round(accuracy, 4))

# Make sure test_data and labels have matching lengths
test_data = test_data.reset_index(drop=True)

# Add actual and predicted labels
test_data["actual_label"] = y_test
test_data["predicted_label"] = predictions

# Convert labels to readable names
test_data["actual"] = test_data["actual_label"].map({
    0: "Fake",
    1: "Real"
})

test_data["predicted"] = test_data["predicted_label"].map({
    0: "Fake",
    1: "Real"
})

# Find incorrect predictions
errors = test_data[
    test_data["actual_label"] != test_data["predicted_label"]
].copy()

# Calculate error rate
error_rate = len(errors) / len(test_data)

print("\n" + "-" * 70)
print("ERROR SUMMARY")
print("-" * 70)

print("Total test samples :", len(test_data))
print("Correct predictions:", len(test_data) - len(errors))
print("Incorrect predictions:", len(errors))
print("Error rate:", round(error_rate * 100, 2), "%")

# Fake predicted as Real
fake_as_real = errors[
    (errors["actual_label"] == 0) &
    (errors["predicted_label"] == 1)
]

# Real predicted as Fake
real_as_fake = errors[
    (errors["actual_label"] == 1) &
    (errors["predicted_label"] == 0)
]

print("\nFake classified as Real:", len(fake_as_real))
print("Real classified as Fake:", len(real_as_fake))

# Save all errors
errors.to_csv(
    "models/error_analysis.csv",
    index=False
)

print("\n" + "-" * 70)
print("SAMPLE MISCLASSIFIED CONTENT")
print("-" * 70)

if len(errors) == 0:
    print("No misclassified samples found.")
else:
    for i, row in errors.head(10).iterrows():

        print("\n" + "=" * 60)
        print("Actual    :", row["actual"])
        print("Predicted :", row["predicted"])

        if "title" in row:
            print("Title     :", str(row["title"])[:300])

        if "content" in row:
            print("Content   :", str(row["content"])[:500])
        elif "text" in row:
            print("Content   :", str(row["text"])[:500])

print("\n" + "=" * 70)
print("STEP 12 COMPLETE")
print("=" * 70)

print("\nError analysis saved to:")
print("models/error_analysis.csv")