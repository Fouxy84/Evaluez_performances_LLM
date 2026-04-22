# Assistant RAG avec Mistral

Ce projet implémente un assistant virtuel basé sur le modèle Mistral, utilisant la technique de Retrieval-Augmented Generation (RAG) pour fournir des réponses précises et contextuelles à partir d'une base de connaissances personnalisée.

## Fonctionnalités

- 🔍 **Recherche sémantique** avec FAISS pour trouver les documents pertinents
- 🤖 **Génération de réponses** avec les modèles Mistral (Small ou Large)
- ⚙️ **Paramètres personnalisables** (modèle, nombre de documents, score minimum)

## Prérequis

- Python 3.9+ 
- Clé API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))

## Installation

1. **Cloner le dépôt**

```bash
git clone <url-du-repo>
cd <nom-du-repo>
```

2. **Créer un environnement virtuel**

```bash
# Création de l'environnement virtuel
python -m venv venv

# Activation de l'environnement virtuel
# Sur Windows
venv\Scripts\activate
# Sur macOS/Linux
source venv/bin/activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Configurer la clé API**

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```
MISTRAL_API_KEY=votre_clé_api_mistral
```

## Structure du projet

```
.
├── MistralChat.py          # Application Streamlit principale
├── indexer.py              # Script pour indexer les documents
├── inputs/                 # Dossier pour les documents sources
├── vector_db/              # Dossier pour l'index FAISS et les chunks
├── database/               # Base de données SQLite pour les interactions
└── utils/                  # Modules utilitaires
    ├── config.py           # Configuration de l'application
    ├── database.py         # Gestion de la base de données
    └── vector_store.py     # Gestion de l'index vectoriel

```

## Utilisation

### 1. Ajouter des documents

Placez vos documents dans le dossier `inputs/`. Les formats supportés sont :
- PDF
- TXT
- DOCX
- CSV
- JSON

Vous pouvez organiser vos documents dans des sous-dossiers pour une meilleure organisation.

### 2. Indexer les documents

Exécutez le script d'indexation pour traiter les documents et créer l'index FAISS :

```bash
python indexer.py
```

Ce script va :
1. Charger les documents depuis le dossier `inputs/`
2. Découper les documents en chunks
3. Générer des embeddings avec Mistral
4. Créer un index FAISS pour la recherche sémantique
5. Sauvegarder l'index et les chunks dans le dossier `vector_db/`

### 3. Lancer l'application

```bash
streamlit run MistralChat.py
```

L'application sera accessible à l'adresse http://localhost:8501 dans votre navigateur.


## Modules principaux

### `utils/vector_store.py`

Gère l'index vectoriel FAISS et la recherche sémantique :
- Chargement et découpage des documents
- Génération des embeddings avec Mistral
- Création et interrogation de l'index FAISS

### `utils/query_classifier.py`

Détermine si une requête nécessite une recherche RAG :
- Analyse des mots-clés
- Classification avec le modèle Mistral
- Détection des questions spécifiques vs générales

### `utils/database.py`

Gère la base de données SQLite pour les interactions :
- Enregistrement des questions et réponses
- Stockage des feedbacks utilisateurs
- Récupération des statistiques

## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `utils/config.py` :
- Modèles Mistral utilisés
- Taille des chunks et chevauchement
- Nombre de documents par défaut
- Nom de la commune ou organisation

---

## Évaluation des performances (RAGAS)

Le projet intègre un pipeline d'évaluation complet basé sur [RAGAS](https://docs.ragas.io/) permettant de mesurer la qualité du système RAG et de comparer les modes de réponse (RAG pur vs routage SQL enrichi).

### Métriques évaluées

| Métrique | Description |
|---|---|
| `faithfulness` | Fidélité de la réponse aux documents récupérés |
| `answer_relevancy` | Pertinence de la réponse par rapport à la question |
| `context_recall` | Couverture du contexte par rapport à la vérité terrain |
| `context_precision` | Précision des documents récupérés |
| `global_score` | Moyenne de faithfulness + answer_relevancy |

### Prérequis — environnement dédié

L'évaluation nécessite un environnement conda séparé (`ragas_env`) avec les dépendances RAGAS :

```bash
conda create -n ragas_env python=3.10 -y
conda activate ragas_env
pip install -r requirements.txt
pip install ragas langchain-mistralai matplotlib
```

### Lancer une évaluation

**Mode baseline (RAG pur)**
```bash
conda run -n ragas_env python evaluate_ragas.py --mode baseline
```

**Mode enriched (routage SQL activé)**
```bash
conda run -n ragas_env python evaluate_ragas.py --mode enriched
```

**Mode comparaison (baseline + enriched en séquence)**
```bash
conda run -n ragas_env python evaluate_ragas.py --mode compare
```

Chaque exécution produit dans `data/` :
- `eval_results_<timestamp>.csv` — résultats détaillés par question (métriques, route, catégorie)
- `eval_summary_<timestamp>.json` — moyennes globales et par catégorie

### Générer le rapport comparatif

Après avoir obtenu deux fichiers CSV (baseline et enriched), générer le rapport avec :

```bash
# En spécifiant explicitement les fichiers à comparer
conda run -n ragas_env python generate_report.py \
  --baseline data/eval_results_<timestamp_baseline>.csv \
  --enriched data/eval_results_<timestamp_enriched>.csv

# Ou sans arguments : sélection automatique des deux derniers fichiers CSV
conda run -n ragas_env python generate_report.py
```

Le rapport est généré dans `data/report/` :

| Fichier | Contenu |
|---|---|
| `01_global_comparison.png` | Barres groupées baseline vs enriched pour chaque métrique |
| `02_score_by_category.png` | Heatmap des scores par catégorie de question |
| `03_delta_per_metric.png` | Graphique waterfall des deltas par métrique |
| `04_route_distribution.png` | Camembert de la distribution RAG / SQL |
| `05_error_flag_by_category.png` | Taux d'erreur par catégorie |
| `rapport_comparatif.txt` | Tableau synthétique + analyse critique des biais |

### Catégories de cas de test

Les 20 cas de test dans `evaluate_ragas.py` couvrent 6 catégories :

| Catégorie | Description |
|---|---|
| `simple` | Questions directes à réponse unique |
| `complex` | Questions multi-critères ou comparatives |
| `ambiguous` | Questions subjectives ou mal définies |
| `noisy` | Questions avec bruit lexical ou hors-sujet |
| `robustness` | Questions de robustesse générale (données en temps réel) |
| `robustness_mixed` | Combinaisons de critères textuels et numériques |

### Résultats de référence

Exécution du 22 avril 2026 (20 questions, modèle `mistral-small`) :

| Métrique | Baseline (RAG) | Enriched (SQL) | Δ |
|---|---|---|---|
| Faithfulness | 0.3705 | 0.2778 | −25% |
| Answer Relevancy | 0.5858 | 0.1023 | −83% |
| Context Recall | 0.0000 | 0.0000 | — |
| Context Precision | 0.0000 | 0.0000 | — |
| **Score global** | **0.4782** | **0.1900** | **−60%** |

> **Interprétation** : le routeur SQL (`_is_sql_question`) utilise des mots-clés trop génériques et envoie l'intégralité des questions en SQL. Le SQL tool retourne des données brutes sans prose narrative, ce qui effondre l'`answer_relevancy`. Les métriques de contexte sont nulles car aucun `TestCase` ne définit de `ground_truth`.

