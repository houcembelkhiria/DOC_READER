# ============================
# 🧠 NLP API - FastAPI Backend
# ============================

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import spacy


# -------------------------------------------------------
# 🚀 Initialisation de l'application FastAPI
# -------------------------------------------------------
app = FastAPI(
    title="NLP API",
    description="API pour le résumé de texte et l'extraction d'entités nommées (NER)",
    version="1.0"
)


# -------------------------------------------------------
# 📘 Chargement du modèle T5 (pour le résumé)
# -------------------------------------------------------
# Modèle T5 pré-entraîné de Hugging Face (petite version pour la rapidité)
MODEL_NAME = "t5-base" # or facebook/bart-large-cnn

# Chargement du tokenizer et du modèle
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Création d’un pipeline Hugging Face pour la tâche de résumé
#summarizer = pipeline("summarization", model=MODEL_NAME)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# -------------------------------------------------------
# 🧩 Chargement du modèle spaCy pour la NER (anglais)
# -------------------------------------------------------
# Pour un modèle français, exécute au préalable :
#   python -m spacy download fr_core_news_md
# puis remplace par :
#   nlp = spacy.load("fr_core_news_md")
nlp = spacy.load("en_core_web_sm")


# -------------------------------------------------------
# 📥 Modèle de donnée d'entrée (Pydantic)
# -------------------------------------------------------
class TextInput(BaseModel):
    text: str


# -------------------------------------------------------
# ✂️ Endpoint : Résumé de texte
# -------------------------------------------------------
@app.post("/summarize")
def summarize(input_data: TextInput):
    """
    Génère un résumé du texte fourni à l'aide du modèle T5.
    """
    text = input_data.text.strip()

    # Appel du pipeline Hugging Face pour le résumé
    summary = summarizer(
        text,
        max_length=100,   # Longueur maximale du résumé
        min_length=20,    # Longueur minimale du résumé
        do_sample=False   # Pas d’échantillonnage aléatoire → résultat stable
    )[0]['summary_text']

    return {"summary": summary}


# -------------------------------------------------------
# 🏷️ Endpoint : Extraction d'entités nommées (NER)
# -------------------------------------------------------
@app.post("/ner")
def ner(input_data: TextInput):
    """
    Extrait les entités nommées (personnes, lieux, organisations, etc.)
    du texte fourni à l’aide de spaCy.
    """
    text = input_data.text.strip()
    doc = nlp(text)

    # Extraction des entités sous forme de dictionnaire
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    return {"entities": entities}
