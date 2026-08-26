from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained model and vectorizer we saved earlier
model = joblib.load("classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Map each category to the department responsible for it
category_to_department = {
    "water": "Water Board",
    "road": "PWD (Public Works Department)",
    "electricity": "Electricity Board",
    "sanitation": "Municipal Sanitation Department"
}

# Create the API application
app = FastAPI()

# Define what the incoming data should look like
class Complaint(BaseModel):
    text: str

# Define the endpoint: when someone sends a complaint here, this function runs
@app.post("/classify")
def classify_complaint(complaint: Complaint):
    X = vectorizer.transform([complaint.text])

    # Get the model's confidence for each possible category
    probabilities = model.predict_proba(X)[0]
    best_index = probabilities.argmax()
    confidence = probabilities[best_index]
    category = model.classes_[best_index]

    # If confidence is too low, mark it for manual review instead of guessing
    CONFIDENCE_THRESHOLD = 0.3
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "category": "Uncategorized",
            "department": "Needs manual review",
            "confidence": round(float(confidence), 2)
        }

    department = category_to_department.get(category, "General")
    return {
        "category": category,
        "department": department,
        "confidence": round(float(confidence), 2)
    }
