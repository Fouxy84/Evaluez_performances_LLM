# utils/config.py
import os
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

# --- Clé API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ Attention: La clé API Mistral (MISTRAL_API_KEY) n'est pas définie dans le fichier .env")
    # Vous pouvez choisir de lever une exception ici ou de continuer avec des fonctionnalités limitées
    # raise ValueError("Clé API Mistral manquante. Veuillez la définir dans le fichier .env")

# --- Modèles Mistral ---
EMBEDDING_MODEL = "mistral-embed"
MODEL_NAME = "mistral-small-latest" # Ou un autre modèle comme mistral-large-latest

# --- Configuration de l'Indexation ---
# INPUT_DATA_URL = os.getenv("INPUT_DATA_URL") # Décommentez si vous utilisez une URL
INPUT_DIR = "inputs"                # Dossier pour les données sources après extraction
VECTOR_DB_DIR = "vector_db"         # Dossier pour stocker l'index Faiss et les chunks
FAISS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss_index.idx")
DOCUMENT_CHUNKS_FILE = os.path.join(VECTOR_DB_DIR, "document_chunks.pkl")

CHUNK_SIZE = 1500                   # Taille des chunks en *caractères* (vise ~512 tokens)
CHUNK_OVERLAP = 150                 # Chevauchement en *caractères*
EMBEDDING_BATCH_SIZE = 32           # Taille des lots pour l'API d'embedding

# --- Configuration de la Recherche ---
SEARCH_K = 5                        # Nombre de documents à récupérer par défaut

# --- Configuration de la Base de Données ---
DATABASE_DIR = "database"
DATABASE_FILE = os.path.join(DATABASE_DIR, "interactions.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}" # URL pour SQLAlchemy

# --- Configuration de l'Application ---
APP_TITLE = "NBA Analyst AI"
NAME = "NBA" # Nom à personnaliser dans l'interface

# --- Paramètres LLM ---
TEMPERATURE = 0.1           # Température de génération (0 = déterministe, 1 = créatif)
TOP_P = 0.9                 # Nucleus sampling (valeur commentée dans le code, conservée pour import)
LLM_CALL_DELAY = 5.0        # Délai (secondes) entre appels LLM successifs (rate limit Mistral)

# --- Monitoring Logfire (optionnel) ---
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN", "")  # Laisser vide pour désactiver l'envoi à Logfire

# --- Mode d'évaluation ---
# 0 = RAG seul (pas de base SQL)
# 1 = Routeur LangGraph (RAG + SQL)
DATABASE_STATUS = 0

# --- Prompt système RAG ---
RAG_SYSTEM_PROMT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Réponds uniquement en te basant sur les informations fournies dans le contexte ci-dessous.
Si la réponse ne figure pas dans le contexte, indique-le clairement sans inventer.

CONTEXTE:
{context_str}

QUESTION:
{question}

RÉPONSE:"""

# --- Questions de test et vérités terrain pour l'évaluation RAGAS ---
QUESTIONS_TEST = [
    # Simple
    "Quel joueur a le meilleur % à 3 points ?",
    "Qui a le plus grand nombre de rebonds cette saison ?",
    # Complex
    "Compare Curry et Durant à 3 points sur les 5 derniers matchs",
    "Quel joueur combine le meilleur TS% et le plus haut nombre de passes décisives ?",
    # Ambiguous
    "Qui est le meilleur joueur ?",
    "Quel est le joueur le plus utile à son équipe cette saison ?",
    # Noisy
    "Quel joueur a le meilleur % avec stats mélangées et inutiles",
    "Donne-moi des stats intéressantes sur n'importe quel joueur NBA",
    # Robustness
    "Combien de victoires ont les Warriors cette saison ?",
    "Compare Curry et Durant sur leurs moyennes de points et de passes",
    "Quel est le nombre de points moyen de LeBron James sur ses 5 derniers matchs ?",
    "Explique pourquoi les Lakers ont plus de victoires que les Suns cette saison.",
    "Quels sont les leaders en pourcentage à 3 points avec au moins 10 tentatives ?",
    "combien de fois Tony Parker a t-il ete MVP durant toutes sa carrière en NBA ?",
    # Robustness mixed
    "Le joueur avec le meilleur PIE cette saison est-il aussi dans le top 5 des scoreurs ?",
    "Parmi les joueurs avec plus de 20 points par match, qui a le meilleur pourcentage aux lancers francs ?",
    "Quel joueur a une moyenne supérieure à 8 passes ET plus de 50% de réussite aux tirs ?",
    "Compare le Net Rating des équipes avec plus de 45 victoires cette saison.",
    "Quel joueur a progressé le plus en pourcentage de réussite à 3 points entre le début et la fin de saison ?",
]

# Vérités terrain associées (None = pas de ground truth défini pour cette question)
GROUND_TRUTHS = [None] * len(QUESTIONS_TEST)