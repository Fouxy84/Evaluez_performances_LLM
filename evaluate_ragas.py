#!/usr/bin/env python3
"""
Evaluation complète d’un système RAG avec RAGAS

Features:
- Batch evaluation (RAGAS best practice)
- Pydantic validation
- Typologie des cas de test
- Logging structuré
- Export CSV + JSON
- Détection automatique des erreurs
- Logfire (optionnel)
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from contextlib import nullcontext
from datetime import datetime
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from pydantic import BaseModel, Field
import faiss
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # Mesure si la réponse est factuellement ancrée dans les contextes récupérés.
                         # Score = proportion d'affirmations de la réponse vérifiables dans le contexte.
                         # Plage : [0, 1] — 1 = entièrement fidèle, 0 = aucune affirmation vérifiable.
    answer_relevancy,    # Mesure si la réponse répond bien à la question posée.
                         # Calculé en générant des questions synthétiques depuis la réponse et en mesurant
                         # leur similarité cosinus avec la question originale.
                         # Plage : [0, 1] — 1 = réponse parfaitement pertinente.
    context_recall,      # Mesure la couverture du contexte récupéré par rapport à la vérité terrain (ground_truth).
                         # Nécessite un ground_truth défini dans le TestCase, sinon retourne NaN/0.
                         # Plage : [0, 1] — 1 = tous les éléments du ground_truth sont dans le contexte.
    context_precision,   # Mesure la proportion de documents récupérés qui sont réellement utiles pour répondre.
                         # Nécessite un ground_truth défini dans le TestCase, sinon retourne NaN/0.
                         # Plage : [0, 1] — 1 = tous les chunks récupérés sont pertinents.
)
from ragas.llms import LangchainLLMWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import Field
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import AIMessage
from sql_tool import create_sql_tool
from utils.database import SessionLocal

class MistralRagasWrapper(BaseChatModel):

    client: Any = Field(...)
    model: str = Field(...)
    
    def _convert_message(self, m):
     role_map = {
         "human": "user",
         "ai": "assistant",
         "system": "system"
         }

     return {
        "role": role_map.get(m.type, "user"),
        "content": m.content
        }

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        temperature = kwargs.get("temperature", 0)
        
        print("DEBUG MESSAGES:", [
        self._convert_message(m) for m in messages])

        response = self.client.chat(
            model=self.model,
           
        messages=[self._convert_message(m) for m in messages])

        content = response.choices[0].message.content

        return ChatResult(
            generations=[
             ChatGeneration(
                 message=AIMessage(content=content)
              )
         ]
         )
    

    @property
    def _llm_type(self):
        return "mistral"

import logfire

from utils.config import MISTRAL_API_KEY, MODEL_NAME
from utils.vector_store import VectorStoreManager
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
import os
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_mistralai.embeddings import MistralAIEmbeddings

mistral_embeddings = LangchainEmbeddingsWrapper(
    MistralAIEmbeddings(model="mistral-embed")
)
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
# ========================
# CONFIG
# ========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USE_LOGFIRE = False

if USE_LOGFIRE:
    logfire.configure()

    def get_span(name):
        return logfire.span(name)
else:
    def get_span(name):
        return nullcontext()


OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


# ========================
# MODELS
# ========================

class QueryInput(BaseModel):
    question: str = Field(..., min_length=1)


class RAGResponse(BaseModel):
    answer: str
    contexts: List[str]


class EvalRow(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]
    category: str
    route: str


# ========================
# TEST CASES
# ========================

@dataclass
class TestCase:
    question: str
    category: str
    ground_truth: Optional[str] = None


TEST_CASES = [
    # --- Simples ---
    TestCase("Quel joueur a le meilleur % à 3 points ?", "simple"),
    TestCase("Qui a le plus grand nombre de rebonds cette saison ?", "simple"),
    # --- Complexes ---
    TestCase("Compare Curry et Durant à 3 points sur les 5 derniers matchs", "complex"),
    TestCase("Quel joueur combine le meilleur TS% et le plus haut nombre de passes décisives ?", "complex"),
    # --- Ambiguës ---
    TestCase("Qui est le meilleur joueur ?", "ambiguous"),
    TestCase("Quel est le joueur le plus utile à son équipe cette saison ?", "ambiguous"),
    # --- Bruyantes ---
    TestCase("Quel joueur a le meilleur % avec stats mélangées et inutiles", "noisy"),
    TestCase("Donne-moi des stats intéressantes sur n'importe quel joueur NBA", "noisy"),
    # --- Robustesse générale ---
    TestCase("Combien de victoires ont les Warriors cette saison ?", "robustness"),
    TestCase("Compare Curry et Durant sur leurs moyennes de points et de passes", "robustness"),
    TestCase("Quel est le nombre de points moyen de LeBron James sur ses 5 derniers matchs ?", "robustness"),
    TestCase("Explique pourquoi les Lakers ont plus de victoires que les Suns cette saison.", "robustness"),
    TestCase("Quels sont les leaders en pourcentage à 3 points avec au moins 10 tentatives ?", "robustness"),
    # --- Robustesse mixte texte + numérique ---
    TestCase(
        "Le joueur avec le meilleur PIE cette saison est-il aussi dans le top 5 des scoreurs ?",
        "robustness_mixed",
    ),
    TestCase(
        "Parmi les joueurs avec plus de 20 points par match, qui a le meilleur pourcentage aux lancers francs ?",
        "robustness_mixed",
    ),
    TestCase(
        "Quel joueur a une moyenne supérieure à 8 passes ET plus de 50% de réussite aux tirs ?",
        "robustness_mixed",
    ),
    TestCase(
        "Compare le Net Rating des équipes avec plus de 45 victoires cette saison.",
        "robustness_mixed",
    ),
    TestCase(
        "Quel joueur a progressé le plus en pourcentage de réussite à 3 points entre le début et la fin de saison ?",
        "robustness_mixed",
    ),
]


# ========================
# UTILS
# ========================

def clean_contexts(contexts: List[str]) -> List[str]:
    return [c.strip() for c in contexts if c and c.strip()]


# ========================
# RAG EVALUATOR
# ========================

class RAGEvaluator:

    def __init__(self):
        self.vs = VectorStoreManager()
        self.client = MistralClient(api_key=MISTRAL_API_KEY)
        self.db = SessionLocal()
        self.sql_tool = create_sql_tool(self.db)

    def _is_sql_question(self, question: str) -> bool:
        sql_keywords = [
            "combien", "quel", "compare", "meilleur", "top",
            "points", "assists", "passes", "moyenne", "victoires",
            "saison", "score", "classement", "leader", "vs"
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in sql_keywords)

    def answer_question(self, question: str) -> str:
        if self.sql_tool and self._is_sql_question(question):
            logger.info("Détection SQL active — utilisation du SQL Tool")
            try:
                response = self.sql_tool.func(question)
                return response
            except Exception as e:
                logger.error(f"Erreur SQL Tool: {e}")
                return "Erreur lors de l'exécution de la requête SQL. Réessayez avec une phrase plus simple."

        logger.info("Fallback RAG — utilisation du modèle de contexte")
        rag = self.query_rag(question)
        return rag.answer

    def get_embedding(self, text: str):
        response = self.client.embeddings(
            model="mistral-embed",
            input=[text]
        )
        return np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

    def query_rag(self, question: str) -> RAGResponse:
        with get_span("rag_query"):

            q = QueryInput(question=question).question
            emb = self.get_embedding(q)
            emb = self.get_embedding(q)
            faiss.normalize_L2(emb)   # 🔥 AJOUT CRITIQUE

            D, I = self.vs.index.search(emb, k=3)

            contexts = [
                self.vs.document_chunks[i].text
                for i in I[0]
                if i < len(self.vs.document_chunks)
            ]
            #debug dis
            print("DISTANCES:", D[0])
            print("INDICES:", I[0])


            contexts = clean_contexts(contexts)

            prompt = f"Contexte:\n{chr(10).join(contexts)}\n\nQuestion: {q}\nRéponse:"
            print('envoie requête au modèle...')
            response = self.client.chat(
                model=MODEL_NAME,
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.1
            )
            print('Réponse reçue du modèle...')
            answer = response.choices[0].message.content

            logger.info(f"\n🧠 QUESTION: {q}")
            logger.info(f"📚 CONTEXTS: {contexts}")
            logger.info(f"💬 ANSWER: {answer}")

            return RAGResponse(answer=answer, contexts=contexts)

    async def build_dataset(self, use_sql_routing: bool = False) -> pd.DataFrame:
        rows = []

        for case in tqdm(TEST_CASES):
            try:
                if use_sql_routing:
                    route = "sql" if self._is_sql_question(case.question) else "rag"
                    answer = self.answer_question(case.question)
                    contexts = [f"Routed via {route} tool"] if route == "sql" else []
                else:
                    rag = self.query_rag(case.question)
                    route = "rag"
                    answer = rag.answer
                    contexts = rag.contexts

                row = EvalRow(
                    question=case.question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=case.ground_truth or "",
                    category=case.category,
                    route=route
                )

                rows.append(row.model_dump())

            except Exception as e:
                logger.error(f"Erreur sur '{case.question}': {e}")

        df = pd.DataFrame(rows)
        df["question"] = df["question"].fillna("").astype(str)
        df["answer"] = df["answer"].fillna("").astype(str)
        df["ground_truth"] = df["ground_truth"].fillna("").astype(str)
        df["category"] = df["category"].fillna("").astype(str)
        df["route"] = df["route"].fillna("").astype(str)
        df["contexts"] = df["contexts"].apply(lambda x: x if isinstance(x, list) else [])

        return df

    def compare_runs(self, baseline_df: pd.DataFrame, enriched_df: pd.DataFrame) -> pd.DataFrame:
        baseline_scores = baseline_df.mean(numeric_only=True).rename("baseline")
        enriched_scores = enriched_df.mean(numeric_only=True).rename("enriched")
        result = pd.concat([baseline_scores, enriched_scores], axis=1)
        result["delta"] = (result["enriched"] - result["baseline"]).round(4)
        return result

    def run_ragas(self, df: pd.DataFrame) -> pd.DataFrame:
        dataset = Dataset.from_pandas(df)
        mistral_llm = MistralRagasWrapper(
            client=self.client,
            model=MODEL_NAME
        )
        llm_wrapper = LangchainLLMWrapper(mistral_llm)

        scores = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision
            ],
            llm=llm_wrapper,
            embeddings=mistral_embeddings,
        )
        scores_df = scores.to_pandas()
         # 🔥 garder uniquement les colonnes utiles
        metric_cols = [
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision"
        ]
        scores_df[metric_cols] = scores_df[metric_cols].fillna(0.0)

        df_final = pd.concat(
            [df.reset_index(drop=True), scores_df[metric_cols]],
        axis=1
        )
        df_final[metric_cols] = df_final[metric_cols].fillna(0.0)

        return df_final


    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        df["global_score"] = df[
            ["faithfulness", "answer_relevancy"]
        ].mean(axis=1, skipna=True).fillna(0.0)

        low_scores = df[["faithfulness", "context_recall"]].fillna(0.0)
        df["error_flag"] = (
            (low_scores["faithfulness"] < 0.6) |
            (low_scores["context_recall"] < 0.6)
        )

        return df

    def save_results(self, df: pd.DataFrame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = OUTPUT_DIR / f"eval_results_{timestamp}.csv"
        json_path = OUTPUT_DIR / f"eval_summary_{timestamp}.json"
        df["category"] = df["category"].apply(lambda x: x[0] if isinstance(x, list) else x)
        df["category"] = df["category"].fillna("").astype(str)
        df["contexts"] = df["contexts"].apply(lambda x: x if isinstance(x, list) else [])
        df["question"] = df["question"].fillna("").astype(str)
        df["answer"] = df["answer"].fillna("").astype(str)
        df["ground_truth"] = df["ground_truth"].fillna("").astype(str)
        df[["faithfulness", "answer_relevancy", "context_recall", "context_precision", "global_score"]]=df[["faithfulness", "answer_relevancy", "context_recall", "context_precision", "global_score"]].fillna(0.0)

        df.to_csv(csv_path, index=False)

        summary = {
            "global_mean": df.mean(numeric_only=True).to_dict(),
            "by_category": df.groupby("category").mean(numeric_only=True).to_dict()
        }

        with open(json_path, "w") as f:
            import json
            json.dump(summary, f, indent=2)

        logger.info(f"📄 CSV saved: {csv_path}")
        logger.info(f"📊 JSON saved: {json_path}")

    async def run(self, use_sql_routing: bool = False) -> pd.DataFrame:
        logger.info("🚀 Building dataset...")
        df = await self.build_dataset(use_sql_routing=use_sql_routing)

        if df.empty:
            logger.error("Dataset vide")
            return df

        logger.info("📊 Running RAGAS...")
        df = self.run_ragas(df)

        df = self.post_process(df)

        self.save_results(df)

        return df

    def build_summary(self, df: pd.DataFrame) -> str:
        metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "global_score"]
        available = [c for c in metric_cols if c in df.columns]

        lines = []

        # Global means
        lines.append("\n--- Moyennes globales ---")
        global_means = df[available].mean().round(4)
        for col, val in global_means.items():
            lines.append(f"  {col:<25} {val:.4f}")

        # Per-category breakdown
        if "category" in df.columns:
            lines.append("\n--- Scores par catégorie ---")
            by_cat = df.groupby("category")[available].mean().round(4)
            lines.append(by_cat.to_string())

        # Error flag summary
        if "error_flag" in df.columns:
            n_errors = int(df["error_flag"].sum())
            n_total = len(df)
            lines.append(f"\n--- Signaux d'erreur : {n_errors}/{n_total} ({100*n_errors/n_total:.0f}%) ---")
            if n_errors > 0:
                error_rows = df[df["error_flag"] == True][["question", "faithfulness", "context_recall"]]
                for _, row in error_rows.iterrows():
                    lines.append(f"  ⚠  {row['question'][:60]}")
                    lines.append(f"     faithfulness={row['faithfulness']:.3f}  context_recall={row['context_recall']:.3f}")

        return "\n".join(lines)


# ========================
# MAIN
# ========================

def main():
    parser = argparse.ArgumentParser(description="Évaluer le système RAG et comparer l'enrichissement SQL.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "enriched", "compare"],
        default="baseline",
        help="Mode d'évaluation à exécuter"
    )
    args = parser.parse_args()

    evaluator = RAGEvaluator()

    if args.mode == "baseline":
        df = asyncio.run(evaluator.run(use_sql_routing=False))
        if not df.empty:
            print("\n=== BASELINE GLOBAL SCORES ===")
            print(evaluator.build_summary(df))
    elif args.mode == "enriched":
        df = asyncio.run(evaluator.run(use_sql_routing=True))
        if not df.empty:
            print("\n=== ENRICHED GLOBAL SCORES ===")
            print(evaluator.build_summary(df))
    else:
        baseline_df = asyncio.run(evaluator.run(use_sql_routing=False))
        enriched_df = asyncio.run(evaluator.run(use_sql_routing=True))

        if baseline_df.empty or enriched_df.empty:
            logger.error("Une des évaluations a échoué, impossible de comparer.")
            return

        print("\n=== BASELINE GLOBAL SCORES ===")
        print(evaluator.build_summary(baseline_df))

        print("\n=== ENRICHED GLOBAL SCORES ===")
        print(evaluator.build_summary(enriched_df))

        compare_df = evaluator.compare_runs(baseline_df, enriched_df)
        print("\n=== COMPARAISON BASELINE vs ENRICHED ===")
        metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "global_score"]
        available = [c for c in metric_cols if c in compare_df.index]
        print(compare_df.loc[available].to_string())

        route_counts = enriched_df["route"].value_counts()
        print("\n=== ROUTAGE DANS L'ENRICHED ===")
        for route, count in route_counts.items():
            pct = 100 * count / len(enriched_df)
            print(f"  {route:<10} {count:>3} questions  ({pct:.0f}%)")

        categories = enriched_df.groupby(["category", "route"]).size().unstack(fill_value=0)
        print("\n=== DISTRIBUTION PAR CATEGORIE ET ROUTE ===")
        print(categories.to_string())

if __name__ == "__main__":
    main()