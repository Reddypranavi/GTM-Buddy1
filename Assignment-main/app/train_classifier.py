import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load data
data = pd.read_csv("calls_dataset.csv")
X = data["text_snippet"]
y = data["labels"].str.get_dummies(",")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = OneVsRestClassifier(LogisticRegression())
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "classifier.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")



# import pandas as pd
# import random

# # Step 1: Generate synthetic dataset
# def generate_dataset():
#     data = []
#     competitors = ["CompetitorX", "CompetitorY", "CompetitorZ"]
#     features = ["analytics", "AI engine", "data pipeline"]
#     pricing_keywords = ["discount", "renewal cost", "budget", "pricing model"]
#     labels_list = [
#         "Positive",
#         "Pricing Discussion",
#         "Objection",
#         "Security",
#         "Competition",
#     ]

#     for i in range(1, 101):  # Generate 100 rows
#         text_snippet = f"We love the {random.choice(features)}, but {random.choice(competitors)} has a cheaper {random.choice(pricing_keywords)}."
#         labels = random.sample(labels_list, random.randint(1, 3))  # Randomly assign 1-3 labels
#         data.append({"id": i, "text_snippet": text_snippet, "labels": ", ".join(labels)})

#     # Save to CSV
#     dataset = pd.DataFrame(data)
#     dataset.to_csv("calls_dataset.csv", index=False)
#     print(dataset)
#     print("calls_dataset.csv generated successfully!")

# # Step 2: Generate dataset
# generate_dataset()

# # Step 3: Load data
# data = pd.read_csv("calls_dataset.csv")
# X = data["text_snippet"]
# y = data["labels"].str.get_dummies(", ")

# # Step 4: Split data
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 5: TF-IDF Vectorization
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer()
# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)

# # Step 6: Model training
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.linear_model import LogisticRegression
# model = OneVsRestClassifier(LogisticRegression())
# model.fit(X_train_vec, y_train)

# # Step 7: Evaluation
# from sklearn.metrics import classification_report
# y_pred = model.predict(X_test_vec)
# print(classification_report(y_test, y_pred))

# # Step 8: Save model and vectorizer
# import joblib
# joblib.dump(model, "classifier.joblib")
# joblib.dump(vectorizer, "vectorizer.joblib")

print("Model and vectorizer saved successfully!")
