"""Production-ready RAG flow leveraging the tuned Elasticsearch hybrid search.

This module builds on top of the ingestion pipeline (``fitness_assistant.ingest``)
and the hyperparameter search performed in ``rag-test-es.py``.  It exposes a small
utility class and CLI for:

* optionally running the ingestion step to (re)populate the Elasticsearch index;
* issuing hybrid keyword + vector queries using the best performing parameters;
* constructing an instruction-focused prompt; and
* generating an answer with OpenAI's GPT models.
"""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from fitness_assistant.ingest import (
    DEFAULT_ES_URL,
    DEFAULT_INDEX_NAME,
    DEFAULT_MODEL_NAME,
    ingest as run_ingest,
)

# --- OpenAI client (replaces Google Gemini) -----------------------------------
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore

# Tuned parameters discovered in `rag-test-es.py`.
BEST_PARAMS: Dict[str, float] = {
    "size": 9,
    "operator": "or",
    "knn_k": 41,
    "num_candidates": 1763,
    "knn_boost": 0.8389895612929347,
    "b_exercise_name": 3.423888532659442,
    "b_muscle_groups": 4.0786853724241805,
    "b_type_of_equipment": 1.730475772103169,
    "b_type_of_activity": 0.9459499781180558,
    "b_body_part": 1.1642902810438727,
    "b_type": 1.3527056181134047,
    "b_instruction": 0.4735048221260146,
}

PROMPT_TEMPLATE = """
You're a professional fitness assistant. Answer the QUESTION based only on the CONTEXT provided from the exercise & fitness database.
- Use only the facts from the CONTEXT when answering the QUESTION.
- If the CONTEXT does not contain the answer, respond with: NONE.
- Keep your answer clear, concise, and detailed with instructions for fitness use.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

ENTRY_TEMPLATE = """
exercise_name: {exercise_name}
type_of_activity: {type_of_activity}
type_of_equipment: {type_of_equipment}
body_part: {body_part}
type: {type}
muscle_groups_activated: {muscle_groups_activated}
instruction: {instruction}
""".strip()


class SentenceTransformerEmbedder:
    """Minimal wrapper so we can reuse the same model for query embeddings."""
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()


class FitnessRAGPipeline:
    """RAG helper that glues together ingestion, retrieval, and answer generation."""
    def __init__(
        self,
        *,
        es_url: str = DEFAULT_ES_URL,
        index_name: str = DEFAULT_INDEX_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
        llm_model: str = "gpt-4.1-mini",
        openai_client: Optional["OpenAI"] = None,
    ) -> None:
        self.es_url = es_url
        self.index_name = index_name
        self.model_name = model_name
        self.llm_model = llm_model

        self._logger = logging.getLogger(self.__class__.__name__)
        self.es = Elasticsearch(self.es_url)
        self.embedder = SentenceTransformerEmbedder(self.model_name)

        # OpenAI client handle
        self._openai_client = openai_client

    # --------------------------------------------------------------------- #
    # Ingestion
    # --------------------------------------------------------------------- #
    def ingest_dataset(
        self,
        csv_path: Path,
        *,
        keep_existing_index: bool = False,
        batch_size: int = 128,
    ) -> None:
        """Delegate to the ingestion pipeline with the current connection details."""
        self._logger.info("Starting ingestion for index '%s'.", self.index_name)
        run_ingest(
            csv_path=csv_path,
            es_url=self.es_url,
            index_name=self.index_name,
            model_name=self.model_name,
            batch_size=batch_size,
            keep_existing_index=keep_existing_index,
        )
        self._logger.info("Ingestion finished for index '%s'.", self.index_name)

    # --------------------------------------------------------------------- #
    # Retrieval
    # --------------------------------------------------------------------- #
    def search(
        self,
        query: str,
        *,
        size: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        params_override: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a hybrid (keyword + kNN) search with tuned defaults.
        ``params_override`` can override any of the keys found in ``BEST_PARAMS``.
        """
        params = dict(BEST_PARAMS)
        if params_override:
            params.update(params_override)
        if size is not None:
            params["size"] = size

        vector = self.embedder.embed_query(query)
        es_bool_filter = self._normalize_filters(filters)
        fields = [
            f"exercise_name^{params['b_exercise_name']}",
            f"muscle_groups_activated^{params['b_muscle_groups']}",
            f"type_of_equipment^{params['b_type_of_equipment']}",
            f"type_of_activity^{params['b_type_of_activity']}",
            f"body_part^{params['b_body_part']}",
            f"type^{params['b_type']}",
            f"instruction^{params['b_instruction']}",
        ]
        body: Dict[str, Any] = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "type": "cross_fields",
                                "operator": params["operator"],
                                "fields": fields,
                            }
                        }
                    ],
                    **({"filter": es_bool_filter} if es_bool_filter else {}),
                }
            },
            "knn": {
                "field": "ei_vector",
                "query_vector": vector,
                "k": int(params["knn_k"]),
                "num_candidates": int(params["num_candidates"]),
                "boost": float(params["knn_boost"]),
                **({"filter": es_bool_filter} if es_bool_filter else {}),
            },
            "size": int(params["size"]),
        }

        self._logger.debug("Hybrid query body: %s", body)
        res = self.es.search(index=self.index_name, body=body)
        hits = res.get("hits", {}).get("hits", [])
        formatted: List[Dict[str, Any]] = []
        for hit in hits:
            src = hit.get("_source", {})
            formatted.append(
                {
                    "id": src.get("id", hit.get("_id")),
                    "exercise_name": src.get("exercise_name", ""),
                    "type_of_activity": src.get("type_of_activity", ""),
                    "type_of_equipment": src.get("type_of_equipment", ""),
                    "body_part": src.get("body_part", ""),
                    "type": src.get("type", ""),
                    "muscle_groups_activated": src.get("muscle_groups_activated", []),
                    "instruction": src.get("instruction", ""),
                }
            )
        return formatted

    @staticmethod
    def _normalize_filters(filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not filters:
            return []
        if "bool" in filters and "filter" in filters["bool"]:
            return list(filters["bool"]["filter"])
        return [filters]

    # --------------------------------------------------------------------- #
    # Prompt construction
    # --------------------------------------------------------------------- #
    def build_context(self, search_results: Sequence[Dict[str, Any]]) -> str:
        blocks = []
        for doc in search_results:
            entry = dict(doc)
            muscles = entry.get("muscle_groups_activated", [])
            if isinstance(muscles, (list, tuple)):
                entry["muscle_groups_activated"] = ", ".join(map(str, muscles))
            blocks.append(ENTRY_TEMPLATE.format_map(entry))
        return "\n\n".join(blocks).strip()

    def build_prompt(self, question: str, search_results: Sequence[Dict[str, Any]]) -> str:
        context = self.build_context(search_results)
        return PROMPT_TEMPLATE.format(question=question, context=context).strip()

    # --------------------------------------------------------------------- #
    # LLM interaction (OpenAI)
    # --------------------------------------------------------------------- #
    def answer(
        self,
        question: str,
        *,
        size: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        params_override: Optional[Dict[str, float]] = None,
        skip_llm: bool = False,
    ) -> Dict[str, Any]:
        """
        Run retrieval (and optionally the LLM) returning answer + supporting context.
        Returns a dict containing ``question``, ``context``, ``documents`` and,
        when ``skip_llm`` is False, the generated ``answer``.
        """
        documents = self.search(
            question,
            size=size,
            filters=filters,
            params_override=params_override,
        )
        context = self.build_context(documents)
        result: Dict[str, Any] = {
            "question": question,
            "context": context,
            "documents": documents,
        }
        if skip_llm:
            return result

        client = self._ensure_openai_client()
        prompt = self.build_prompt(question, documents)

        # Using Chat Completions for wide compatibility with gpt-4.1-mini
        resp = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a professional fitness assistant. Follow instructions exactly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        answer_text = (resp.choices[0].message.content or "").strip()
        result["answer"] = answer_text
        return result

    def _ensure_openai_client(self) -> "OpenAI":
        if OpenAI is None:  # pragma: no cover - runtime safeguard
            raise RuntimeError(
                "openai is not installed. Install it with: pip install --upgrade openai"
            )
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set; cannot call OpenAI."
            )
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

_PIPELINE: Optional[FitnessRAGPipeline] = None
def get_pipeline(**kwargs: Any) -> FitnessRAGPipeline:
    """
    Return a module-level singleton pipeline, optionally configuring it on first use.
    Subsequent calls ignore kwargs; create a new pipeline manually if you need different
    parameters.
    """
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = FitnessRAGPipeline(**kwargs)
    return _PIPELINE
def rag(
    question: str,
    *,
    size: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    params_override: Optional[Dict[str, float]] = None,
    skip_llm: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper around :class:`FitnessRAGPipeline.answer` for simple scripts.
    """
    pipeline = get_pipeline()
    return pipeline.answer(
        question,
        size=size,
        filters=filters,
        params_override=params_override,
        skip_llm=skip_llm,
    )

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        "-q",
        help="Question to send through the RAG flow. If omitted you'll be prompted.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run the ingestion pipeline before querying.",
    )
    parser.add_argument(
        "--keep-existing-index",
        action="store_true",
        help="When ingesting, do not delete/recreate the index if it already exists.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "fitness_exercises_500.csv",
        help="Dataset to ingest when --ingest is provided.",
    )
    parser.add_argument(
        "--es-url",
        default=DEFAULT_ES_URL,
        help=f"Elasticsearch URL (default: {DEFAULT_ES_URL}).",
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help=f"Elasticsearch index name (default: {DEFAULT_INDEX_NAME}).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"SentenceTransformer model name (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="OpenAI model identifier to use when generating the answer (default: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Chunk size for bulk indexing during ingestion.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=int(BEST_PARAMS["size"]),
        help="Number of documents to retrieve (defaults to tuned value).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip calling OpenAI; only print retrieved context.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    pipeline = FitnessRAGPipeline(
        es_url=args.es_url,
        index_name=args.index_name,
        model_name=args.model_name,
        llm_model=args.llm_model,
    )

    if args.ingest:
        pipeline.ingest_dataset(
            csv_path=args.csv_path,
            keep_existing_index=args.keep_existing_index,
            batch_size=args.batch_size,
        )

    question = args.question or input("Enter your fitness question: ").strip()
    result = pipeline.answer(
        question,
        size=args.size,
        skip_llm=args.no_llm,
    )

    if args.no_llm:
        print(result["context"])
    else:
        print(result.get("answer", ""))


if __name__ == "__main__":
    main()
