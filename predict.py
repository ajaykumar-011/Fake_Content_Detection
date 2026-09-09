import joblib

MODEL_PATH = "models/best_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading TF-IDF vectorizer...")
vectorizer = joblib.load(VECTORIZER_PATH)

print("\n" + "=" * 60)
print("       FAKE CONTENT DETECTION - PREDICTION TEST")
print("=" * 60)

while True:
    text = input("\nEnter news/content (type 'exit' to stop): ").strip()

    if text.lower() == "exit":
        print("\nPrediction testing completed.")
        break

    if not text:
        print("Please enter some content.")
        continue

    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    if prediction == 0:
        result = "FAKE"
    else:
        result = "REAL"

    print("\n" + "-" * 40)
    print("PREDICTION:", result)
    print("-" * 40)