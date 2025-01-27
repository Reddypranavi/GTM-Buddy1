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
This project implements a text classification and entity extraction system with the following features:

Model Development: Train and evaluate models with advanced text preprocessing techniques.
Inference Pipeline: Accepts text snippets and returns classification labels and extracted entities via a REST API or CLI tool.
Scalability and Portability: Fully dockerized for easy deployment.
Setup Instructions
Prerequisites
Install Docker if using the Dockerized setup.
Install Python 3.8+ and pip for local setup.
Clone the Repositry
git clone https://github.com/your-username/text-classification.git
cd text-classification
Install Dependencies
python -m venv venv
venv\Scripts\activate
pip install -r app/requirements.txt
Run the API locally
cd app
python train_classifier.py
uvicorn main:app --host 0.0.0.0 --port 5000
Docker Setup
Build the docker image
docker build -t app .
Run the docker container
docker run -p 5000:5000 app 
API Usage:    Local: http://127.0.0.1:5000/predict
Sample Input
json
{
    "text": "We love the analytics, but CompetitorX has a cheaper subscription"
}
Sample Output
json
{
    "Extracted Entities": {
        "features": ["analytics"]
    },
    "Predicted Labels": ["Security"],
    "Summary": "The text snippet discusses Security."
}
