import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from entity_extraction import extract_entities
from model import classify_text
from utiles import summarize_text

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from entity_extraction import extract_entities


# Initialize FastAPI app
app = FastAPI()

# Load pre-trained model and vectorizer
model = joblib.load("classifier.joblib")
vectorizer = joblib.load("vectorizer.joblib")

class TextSnippet(BaseModel):
    text: str

@app.post("/predict")
def predict(snippet: TextSnippet):
    text = snippet.text
    # Classification
    labels = classify_text(text, model, vectorizer)
    # Entity extraction
    entities = extract_entities(text)
    # Summarization
    summary = summarize_text(text)
    # Return results
    return {"labels": labels, "entities": entities, "summary": summary}
