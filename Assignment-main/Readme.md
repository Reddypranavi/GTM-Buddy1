project/
├── app/
│   ├── main.py              # REST API for inference
│   ├── entity_extraction.py # Entity extraction logic
│   ├── model.py             # Classification model logic
│   ├── utils.py             # Helper functions (e.g., preprocessing)
│   ├── domain_knowledge.json # Knowledge base for entity extraction
│   ├── classifier.joblib    # Pre-trained classification model (exported from Task 1)
│   ├── vectorizer.joblib    # Pre-trained TF-IDF vectorizer (exported from Task 1)
│   ├── train_classifier.py  # Training script for Task 1
│   └── requirements.txt     # Python dependencies
├── Dockerfile               # Docker configuration
├── README.md                # Project setup and instructions
└── run.sh                   # Shell script to build and run the service


1.python -m venv venv
2.venv\Scripts\activate
3.pip install -r app/requirements.txt
4.cd app
5.python train_classifier.py
6.uvicorn main:app --reload
7.http://127.0.0.1:8000/docs

8.curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text": "We love analytics, but CompetitorX offers cheaper pricing."}'
