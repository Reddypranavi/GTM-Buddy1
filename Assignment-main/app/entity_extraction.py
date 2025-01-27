import json
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    with open("domain_knowledge.json") as f:
        knowledge_base = json.load(f)

    # Dictionary-based extraction
    competitors = [word for word in knowledge_base["competitors"] if word in text]
    features = [word for word in knowledge_base["features"] if word in text]
    pricing = [word for word in knowledge_base["pricing_keywords"] if word in text]

    # NER-based extraction
    doc = nlp(text)
    ner_entities = [(ent.text, ent.label_) for ent in doc.ents]

    return {
        "competitors": competitors,
        "features": features,
        "pricing_keywords": pricing,
        "ner_entities": ner_entities,
    }
