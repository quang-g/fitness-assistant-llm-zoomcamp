"""Ingest the fitness exercise dataset into Elasticsearch with vector embeddings."""

from __future__ import annotations  # must come first

# stdlib
import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

# third-party
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


# Default configuration pulled from the exploratory ingestion notebook.
DEFAULT_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
DEFAULT_INDEX_NAME = "500_fitness_exercises"
DEFAULT_ES_URL = os.getenv("ES_URL", "http://localhost:9200")
EMBED_DIMENSIONS = 384
DATA_PATH = os.getenv("DATA_PATH", "../data/fitness_exercises_500.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(DATA_PATH),
        help="Path to the CSV containing the exercises.",
    )
    parser.add_argument(
        "--es-url",
        default=DEFAULT_ES_URL,
        help="Elasticsearch URL, e.g. http://localhost:9200",
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help="Name of the Elasticsearch index to recreate.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model to use for embedding generation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of documents per bulk indexing request.",
    )
    parser.add_argument(
        "--keep-existing-index",
        action="store_true",
        help="Skip deleting/recreating the index if it already exists.",
    )
    return parser.parse_args()


def load_documents(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "muscle_groups_activated" in df.columns:
        df["muscle_groups_activated"] = (
            df["muscle_groups_activated"]
            .fillna("")
            .apply(
                lambda value: [
                    group.strip()
                    for group in str(value).split(",")
                    if group and group.strip()
                ]
            )
        )

    df = df.fillna("")
    documents = df.to_dict(orient="records")
    return documents


def embed_documents(documents: Sequence[dict], model: SentenceTransformer) -> None:
    for doc in tqdm(documents, desc="Embedding documents"):
        exercise_name = str(doc.get("exercise_name", "") or "")
        instruction = str(doc.get("instruction", "") or "")
        combined = f"{exercise_name} {instruction}".strip()

        doc["exercise_name_vector"] = model.encode(exercise_name).tolist()
        doc["instruction_vector"] = model.encode(instruction).tolist()
        doc["ei_vector"] = model.encode(combined).tolist()


def recreate_index(
    es_client: Elasticsearch,
    index_name: str,
    *,
    keep_existing: bool,
) -> None:
    if not keep_existing:
        es_client.indices.delete(index=index_name, ignore_unavailable=True)

    if keep_existing and es_client.indices.exists(index=index_name):
        return

    body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "normalizer": {
                    "lc": {
                        "type": "custom",
                        "char_filter": [],
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            },
        },
        "mappings": {
            "properties": {
                "exercise_name": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword", "normalizer": "lc"}},
                },
                "type_of_activity": {
                    "type": "keyword",
                    "normalizer": "lc",
                    "fields": {"text": {"type": "text"}},
                },
                "type_of_equipment": {
                    "type": "keyword",
                    "normalizer": "lc",
                    "fields": {"text": {"type": "text"}},
                },
                "body_part": {"type": "keyword", "normalizer": "lc"},
                "type": {"type": "keyword", "normalizer": "lc"},
                "muscle_groups_activated": {"type": "keyword", "normalizer": "lc"},
                "instruction": {"type": "text"},
                "exercise_name_vector": {
                    "type": "dense_vector",
                    "dims": EMBED_DIMENSIONS,
                    "index": True,
                    "similarity": "cosine",
                },
                "instruction_vector": {
                    "type": "dense_vector",
                    "dims": EMBED_DIMENSIONS,
                    "index": True,
                    "similarity": "cosine",
                },
                "ei_vector": {
                    "type": "dense_vector",
                    "dims": EMBED_DIMENSIONS,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }

    es_client.indices.create(index=index_name, body=body)


def bulk_actions(documents: Sequence[Mapping], index_name: str) -> Iterable[Mapping]:
    for doc in documents:
        doc_id = doc.get("id")
        yield {
            "_index": index_name,
            "_id": doc_id,
            "_op_type": "index",
            "_source": doc,
        }


def ingest(
    *,
    csv_path: Path,
    es_url: str,
    index_name: str,
    model_name: str,
    batch_size: int,
    keep_existing_index: bool,
) -> None:
    logger = logging.getLogger("ingest")
    logger.info("Loading dataset from %s", csv_path)
    documents = load_documents(csv_path)
    logger.info("Loaded %d documents", len(documents))

    logger.info("Loading embedding model %s", model_name)
    model = SentenceTransformer(model_name)

    embed_documents(documents, model)

    logger.info("Connecting to Elasticsearch at %s", es_url)
    es_client = Elasticsearch(es_url)

    recreate_index(es_client, index_name, keep_existing=keep_existing_index)
    if not keep_existing_index:
        logger.info("Index %s recreated.", index_name)
    else:
        logger.info("Index %s ready.", index_name)

    logger.info("Indexing documents in batches of %d", batch_size)
    helpers.bulk(
        es_client,
        bulk_actions(documents, index_name),
        chunk_size=batch_size,
        request_timeout=120,
    )
    logger.info("Ingestion complete.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()

    ingest(
        csv_path=args.csv_path,
        es_url=args.es_url,
        index_name=args.index_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
        keep_existing_index=args.keep_existing_index,
    )


if __name__ == "__main__":
    main()
