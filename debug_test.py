import joblib

model = joblib.load("classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

test_sentence = "no water for 3 days"

X = vectorizer.transform([test_sentence])
probabilities = model.predict_proba(X)[0]

print(f"\nTesting: '{test_sentence}'\n")
for category, prob in zip(model.classes_, probabilities):
    print(f"{category}: {round(prob * 100, 1)}%")