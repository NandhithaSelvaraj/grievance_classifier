from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

from train_data import complaints, categories

# Step 1: Convert sentences into numbers the model can understand
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(complaints)

# Step 2: Train the model to learn the pattern between text and category
model = SVC(probability=True)
model.fit(X, categories)

# Step 3: Save the trained model and vectorizer to files
joblib.dump(model, "classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved successfully!")