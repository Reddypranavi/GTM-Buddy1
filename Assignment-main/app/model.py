def classify_text(text, model, vectorizer):
    vectorized_text = vectorizer.transform([text])
    predictions = model.predict(vectorized_text)
    return predictions[0].tolist()
