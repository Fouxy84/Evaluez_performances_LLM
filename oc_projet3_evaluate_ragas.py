# evaluate_ragas.py pour lancer une évaluation rapide du processus RAG+LLM
import os
import sys
import time
import asyncio
import nest_asyncio  # Permet d'imbriquer des event loops (utile en notebook)
import warnings
import logfire
import httpx
from dataclasses import dataclass
from mistralai import SDKError

# --- Importations depuis vos modules ---
from utils.vector_store import VectorStoreManager
from utils.config import (
    MISTRAL_API_KEY, MODEL_NAME, TEMPERATURE, TOP_P, EMBEDDING_MODEL,
    SEARCH_K, QUESTIONS_TEST, GROUND_TRUTHS, RAG_SYSTEM_PROMT, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_BATCH_SIZE,
    LOGFIRE_TOKEN, DATABASE_STATUS, LLM_CALL_DELAY,
)

# Import conditionnel du graphe routeur (uniquement si la base de données sql est activée via config.py)
if DATABASE_STATUS == 1:
    from utils.langgraph_app import build_graph

# --- Imports RAGAS 0.4.x ---
from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._answer_relevance import ResponseRelevanceInput
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding
from langchain_core.outputs import Generation, LLMResult

# Autorise l'imbrication de boucles asyncio (nécessaire dans les notebooks Jupyter)
nest_asyncio.apply()

# --- Configuration de Logfire ---
if LOGFIRE_TOKEN:
    logfire.configure(token=LOGFIRE_TOKEN, send_to_logfire=True)
else:
    logfire.configure(send_to_logfire=False)  # Mode silencieux si pas de token


# --- Définition du type d'évaluation ---

eval_type = ['_initial', '_final']
if DATABASE_STATUS == 0:
    eval_select = eval_type[0]   # Évaluation RAG seul
else:
    eval_select = eval_type[1]   # Évaluation du routeur (RAG et SQL)

# ===========================================================
# Réécriture du client Ragas pour Mistral natif (sur la base de BaseRagasLLM)
# ===========================================================

@dataclass
class MistralRagasLLM(BaseRagasLLM):
    """BaseRagasLLM natif qui réutilise le client Mistral du VectorStoreManager."""
    model: str = MODEL_NAME
    temperature: float = TEMPERATURE
    # top_p: float = TOP_P
    multiple_completion_supported: bool = True  # Permet de générer n réponses en un appel

    # Référence au client partagé (injecté après instanciation du VectorStoreManager)
    _client: object = None

    def __post_init__(self):
        super().__post_init__()
        # _client sera injecté via set_client() après init du VectorStoreManager

    def set_client(self, client):
        """Injecte le client Mistral partagé (depuis VectorStoreManager.mistral_client)."""
        self._client = client

    def _call_api(self, prompt_text: str, n: int = 1, temperature: float = 0.01
                  #, top_p: float = TOP_P
                  ) -> list:
        """Appels synchrones à l'API Mistral, retourne une liste de textes.
        Boucle n fois pour générer plusieurs completions (utilisé par AnswerRelevancy).
        Intègre un retry avec backoff exponentiel en cas d'erreur 429 (rate limit)."""
        results = []
        for i in range(n):
            if i > 0:
                time.sleep(LLM_CALL_DELAY)  # Délai entre completions successives pour le même prompt
            max_retries = 6
            wait_time = 10  # Attente en cas d'erreur
            for attempt in range(max_retries):
                try:
                    resp = self._client.chat.complete(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=temperature,
                        #top_p=top_p,
                    )
                    results.append(resp.choices[0].message.content)
                    break  # Succès : on sort de la boucle de retry
                except SDKError as e:
                    if getattr(e, "status_code", None) == 429 and attempt < max_retries - 1:
                        logfire.warning(
                            f"Rate limit 429 dans _call_api (tentative {attempt + 1}/{max_retries}). "
                            f"Attente {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        wait_time = min(wait_time * 2, 120)  # Backoff exponentiel, max 120s
                    else:
                        logfire.error(f"SDKError non-récupérable dans _call_api: {e}")
                        raise
        return results

    def generate_text(self, prompt, n=1, temperature=0.01
                      #, top_p=TOP_P
                        , stop=None, callbacks=None) -> LLMResult:
        # Appel synchrone : convertit le prompt RAGAS en LLMResult LangChain
        texts = self._call_api(prompt.to_string(), n=n, temperature=temperature
                               #, top_p=top_p
                                )
        generations = [[Generation(text=t) for t in texts]]
        return LLMResult(generations=generations)

    async def agenerate_text(self, prompt, n=1, temperature=0.01
                             #, top_p=TOP_P
                            , stop=None, callbacks=None) -> LLMResult:
        # Appel asynchrone : délègue l'appel bloquant à un thread pour ne pas geler l'event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self._call_api(prompt.to_string(), n=n, temperature=temperature
                                         #, top_p=top_p
                                         )
        )
        generations = [[Generation(text=t) for t in result]]
        return LLMResult(generations=generations)

    def is_finished(self, response: LLMResult) -> bool:
        # L'API Mistral renvoie toujours une réponse complète, pas de streaming partiel
        return True

# ===========================================================
# Réécriture du client Ragas pour Mistral embeddings natif (sur la base de BaseRagasEmbedding)
# ===========================================================

class MistralRagasEmbeddings(BaseRagasEmbedding):
    """BaseRagasEmbedding natif qui délègue à _generate_embeddings() du VectorStoreManager
    via son mistral_client partagé, évitant la création d'un second client Mistral."""

    def __init__(self, client, model: str = EMBEDDING_MODEL):
        super().__init__()
        self.model = model
        self._client = client  # client partagé : VectorStoreManager.mistral_client

    def embed_text(self, text: str, **kwargs) -> list:
        # Embedding d'un texte unique via l'API Mistral
        response = self._client.embeddings.create(model=self.model, inputs=[text])
        return response.data[0].embedding

    async def aembed_text(self, text: str, **kwargs) -> list:
        # Version asynchrone : exécute embed_text dans un thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text)

    def embed_texts(self, texts: list, **kwargs) -> list:
        # Embedding batch : traite plusieurs textes en un seul appel API
        response = self._client.embeddings.create(model=self.model, inputs=texts)
        return [item.embedding for item in response.data]

    # Aliases attendus par ResponseRelevancy.calculate_similarity
    def embed_query(self, text: str) -> list:
        return self.embed_text(text)

    def embed_documents(self, texts: list) -> list:
        return self.embed_texts(texts)

# ===========================================================
# Personnalisation de la classe Ragas AnswerRelevancy pour éviter le blocage de l'event loop asyncio
# ===========================================================

# --- Sous-classe d'AnswerRelevancy avec calculate_similarity non-bloquant ---
class AsyncAnswerRelevancy(AnswerRelevancy):
    """Version d'AnswerRelevancy dont calculate_similarity s'exécute dans un thread
    pour ne pas bloquer l'event loop asyncio."""

    async def _ascore(self, row: dict, callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        # Génère `strictness` questions candidates à partir de la réponse du LLM
        prompt_input = ResponseRelevanceInput(response=row["response"])
        responses = await self.question_generation.generate_multiple(
            data=prompt_input, llm=self.llm, callbacks=callbacks, n=self.strictness
        )
        # Calcul du score dans un thread pour éviter le blocage de l'event loop
        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(None, self._calculate_score, responses, row)
        return score

# ===========================================================
# Initialisation des variables, du Vector Store et du graphe routeur (si DATABASE_STATUS == 1)
# ===========================================================

# --- Vérification de la clé API Mistral ---
if not MISTRAL_API_KEY:
    logfire.error("Erreur : Clé API Mistral non trouvée (MISTRAL_API_KEY). Veuillez la définir dans le fichier .env.")
    sys.exit(1)


# --- Stockage des résultats pour RAGAS ---
placeholder_contexts = []  # Contextes récupérés par le RAG (liste de listes)
answers = []               # Réponses générées par le LLM

# --- Fonctions ---

def get_vector_store_manager():
    """Essaie de charger le VectorStoreManager et son index. Retourne None en cas d'échec."""
    logfire.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        if manager.index is None or not manager.document_chunks:
            logfire.error("Index Faiss ou chunks non trouvés/chargés par VectorStoreManager.")
            return None
        logfire.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
        return manager
    except FileNotFoundError:
        logfire.error("FileNotFoundError lors de l'init de VectorStoreManager.")
        return None
    except Exception as e:
        logfire.exception("Erreur chargement VectorStoreManager")
        return None


# Initialisation du Vector Store Manager
vector_store_manager = get_vector_store_manager()

# Avertissement si l'index ne contient pas de données Excel (pertinent en mode DATABASE_STATUS==0)
if vector_store_manager and DATABASE_STATUS == 0:
    has_excel = any(
        ".xlsx" in (c.get("metadata", {}).get("source", "") if isinstance(c, dict) else "")
        for c in (vector_store_manager.document_chunks or [])
    )
    if not has_excel:
        logfire.warning(
            "⚠️  DATABASE_STATUS==0 mais aucun chunk Excel trouvé dans l'index FAISS. "
            "Les métriques RAGAS sur les questions statistiques risquent d'être à 0. "
            "Relancez indexer.py avec DATABASE_STATUS=0 pour inclure les données Excel."
        )

# Initialisation du graphe routeur (DATABASE_STATUS == 1 uniquement)
router_graph = None
if DATABASE_STATUS == 1:
    if vector_store_manager is None:
        logfire.error("VectorStoreManager requis même en mode routeur (branche RAG + client Mistral RAGAS).")
        sys.exit(1)
    router_graph, _ = build_graph(vector_store_manager)
    logfire.info("Graphe routeur LangGraph initialisé (mode évaluation routeur).")

# Suivi des routes empruntées (pour colonne CSV en mode routeur)
router_routes: list = []

def invoke_graph_with_retry(graph, inputs: dict, max_retries: int = 6, initial_wait: float = 15.0) -> dict:
    """Invoque le graphe LangGraph avec retry+backoff exponentiel sur les 429 (rate limit).

    Args:
        graph       : le graphe compilé LangGraph à invoquer.
        inputs      : dictionnaire d'entrée passé à graph.invoke().
        max_retries : nombre maximal de tentatives (défaut : 6).
        initial_wait: attente initiale en secondes avant le premier retry (défaut : 15s).

    Returns:
        Le résultat de graph.invoke() en cas de succès.

    Raises:
        httpx.HTTPStatusError : si toutes les tentatives ont échoué avec 429.
        Exception             : pour toute autre erreur non-récupérable.
    """
    wait_time = initial_wait
    for attempt in range(max_retries):
        try:
            return graph.invoke(inputs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                logfire.warning(
                    f"Rate limit 429 dans invoke_graph_with_retry "
                    f"(tentative {attempt + 1}/{max_retries}). Attente {wait_time:.0f}s..."
                )
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, 120)  # Backoff exponentiel, max 120 s
            else:
                raise  # 429 épuisé ou autre code HTTP → remonte l'exception
        except Exception as e:
            # Pour toute autre exception non-récupérable, on sort immédiatement
            raise
    # Ne devrait jamais être atteint, mais sécurité
    raise RuntimeError("invoke_graph_with_retry : nombre maximal de tentatives dépassé.")

# ===========================================================
# Interrogation du RAG+LLM ou du RAL+SQL+LLM et récupération si pertinent du contexte
# ===========================================================

# --- Boucle sur les questions de test pour générer les réponses ---
# Chaque question est soumise au pipeline (RAG ou routeur) pour collecter
# la réponse et les contextes utilisés, qui alimenteront l'évaluation RAGAS.
for idx_q, (question_test, ground_truth) in enumerate(zip(QUESTIONS_TEST, GROUND_TRUTHS)):
    if idx_q > 0:
        time.sleep(LLM_CALL_DELAY * 3)  # Délai entre chaque question pour respecter le rate limit Mistral
    prompt = question_test
    logfire.info(f"Question test: '{prompt}'")
    logfire.info(f"Ground truth attendue: '{ground_truth}'")

    if DATABASE_STATUS == 0:
        # ── Mode RAG seul ────────────────────────────────────────────────────
        if vector_store_manager is None:
            logfire.error("VectorStoreManager non disponible pour la recherche.")
            sys.exit(1)

        try:
            logfire.info(f"Recherche de contexte pour la question: '{prompt}' avec k={SEARCH_K}")
            search_results = vector_store_manager.search(prompt, k=SEARCH_K)
            logfire.info(f"{len(search_results)} chunks trouvés dans le Vector Store.")
        except Exception as e:
            logfire.exception(f"Erreur pendant vector_store_manager.search pour la query: {prompt}")
            search_results = []

        # Extraction du texte brut depuis chaque chunk retourné
        contexts_list = [res['text'] for res in search_results] if search_results else [""]

        answer = vector_store_manager.generate_rag_response(
            question=prompt,
            context_results=search_results,
            system_prompt_template=RAG_SYSTEM_PROMT,
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            #top_p=TOP_P,
        )

    else:
        # ── Mode Routeur LangGraph (RAG ou SQL) ──────────────────────────────
        try:
            logfire.info(f"[Eval Routeur] Invocation du graphe pour : '{prompt}'")
            result = invoke_graph_with_retry(router_graph, {"user_question": prompt})
            answer = result.get("final_answer", "")
            route  = result.get("route", "?")
            router_routes.append(route)
            logfire.info(f"[Eval Routeur] route={route} | réponse={answer[:80]}...")

            if route == "rag":
                # Route RAG : chunks FAISS récupérés → métriques contextuelles classiques
                contexts_list = result.get("rag_contexts", [""])
            else:
                # TODO : voir si c'est bien utile  de mettre dans le contexte la réponse pour Faithfullness ?
                # Route SQL : la réponse de l'agent SQL constitue le "contexte récupéré"
                # (donnée extraite de la BDD, analogue aux chunks RAG pour les métriques RAGAS)
                contexts_list = [answer] if answer else ["(aucune donnée SQL disponible)"]
                logfire.info(f"[Eval Routeur] Contexte RAGAS (SQL) = réponse SQL ({len(answer)} car.)")

        except Exception as e:
            logfire.exception(f"Erreur lors de l'invocation du graphe routeur pour : '{prompt}'")
            answer        = "Erreur : impossible de traiter la requête."
            contexts_list = [answer]
            router_routes.append("error")

    placeholder_contexts.append(contexts_list)
    answers.append(answer)

# ===========================================================
# Evaluation RAGAS
# ===========================================================

# --- Création du dataset d'évaluation RAGAS (format natif 0.4.x) ---
# Chaque SingleTurnSample regroupe la question, la réponse, les contextes et la référence attendue.
eval_samples = [
    SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth,
    )
    for question, answer, contexts, ground_truth in zip(
        QUESTIONS_TEST, answers, placeholder_contexts, GROUND_TRUTHS
    )
]
evaluation_dataset = EvaluationDataset(samples=eval_samples)
logfire.info(f"Dataset d'évaluation créé avec {len(evaluation_dataset)} exemples.")

# --- Évaluation RAGAS ---
try:
    # RunConfig avec max_workers=1 pour respecter le rate limit Mistral (éviter les 429)
    run_config = RunConfig(
        max_workers=1,    # 1 requête à la fois → pas de 429 Too Many Requests
        max_retries=10,   # 10 tentatives en cas d'erreur temporaire
        max_wait=120,     # Attente max de 120s entre retries (backoff exponentiel)
        timeout=180,      # Timeout de 180s par requête
    )

    # 1. Instancier le LLM et les Embeddings natifs RAGAS en réutilisant
    #    le client Mistral partagé du VectorStoreManager (pas de second client)
    logfire.info("Initialisation du LLM ragas (MistralRagasLLM)...")
    ragas_llm = MistralRagasLLM(model=MODEL_NAME, temperature=TEMPERATURE
                                #, top_p=TOP_P
                                )
    ragas_llm.set_client(vector_store_manager.mistral_client)
    ragas_llm.set_run_config(run_config)
    logfire.info("LLM ragas initialisé.")

    logfire.info("Initialisation des embeddings (MistralRagasEmbeddings)...")
    ragas_embeddings = MistralRagasEmbeddings(client=vector_store_manager.mistral_client)
    logfire.info("Embeddings initialisés.")

    # 2. Définir les métriques et les initialiser avec le RunConfig
    metrics_to_evaluate = [
        Faithfulness(),        # Le LLM reste-t-il fidèle aux contextes fournis ?
        AsyncAnswerRelevancy(),# La réponse est-elle pertinente par rapport à la question ?
        ContextPrecision(),    # Les chunks récupérés sont-ils tous utiles ?
        ContextRecall(),       # Les chunks couvrent-ils bien la réponse attendue ?
    ]
    for m in metrics_to_evaluate:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_embeddings
        m.init(run_config)   # Propagation du RunConfig à chaque métrique

    logfire.info(f"Métriques sélectionnées: {[m.name for m in metrics_to_evaluate]}")

    # 3. Lancer l'évaluation RAGAS
    print("\nLancement de l'évaluation Ragas ...")
    logfire.info("Lancement de l'évaluation Ragas ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        results = evaluate(
            dataset=evaluation_dataset,
            metrics=metrics_to_evaluate,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
        )
    print("\n--- Évaluation Ragas terminée ---")
    logfire.info("Évaluation Ragas terminée.")

    # 4. Afficher les résultats
    print("\n--- Résultats (moyenne)---")
    print(results)

    # 5. Enrichissement des métriques avec les paramètres de configuration et sauvegarde dans un CSV (mode append)
    try:
        results_df = results.to_pandas()
        # Ajout des paramètres de configuration comme colonnes (traçabilité des runs)
        results_df["database_status"]      = DATABASE_STATUS
        results_df["model_name"]           = MODEL_NAME
        results_df["temperature"]          = TEMPERATURE
        # results_df["top_p"]                = TOP_P
        results_df["embedding_model"]      = EMBEDDING_MODEL
        results_df["search_k"]             = SEARCH_K
        results_df["chunk_size"]           = CHUNK_SIZE
        results_df["chunk_overlap"]        = CHUNK_OVERLAP
        results_df["embedding_batch_size"] = EMBEDDING_BATCH_SIZE

        # Colonne route uniquement en mode routeur (rag / sql / error)
        if DATABASE_STATUS == 1 and router_routes:
            results_df["route"] = router_routes

        # Sauvegarde en mode append : permet d'accumuler les runs successifs dans un seul fichier
        output_csv = './Ragas_results/ragas_results' + eval_select + '.csv'
        write_header = not os.path.exists(output_csv)  # En-tête uniquement à la première écriture
        results_df.to_csv(output_csv, mode="a", header=write_header, index=False, sep=';', encoding='utf-8')
        print(f"\nRésultats sauvegardés dans '{output_csv}' (mode append).")
        logfire.info(f"Résultats sauvegardés dans '{output_csv}' (mode append).")

    except AttributeError:
        logfire.warn("Format de résultat non convertible directement en DataFrame Pandas.")

except Exception as e:
    logfire.exception(f"Une erreur est survenue lors de l'initialisation ou l'évaluation : {e}")
