"""
Indexation des passages dans Qdrant (local) avec metadonnees.

Entrees : data/processed/passages.jsonl, data/processed/metadata.csv
Sortie  : base Qdrant dans data/qdrant, collection kb_passages



"""
import argparse
import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
QDRANT_PATH = ROOT / "data" / "qdrant"
COLLECTION = "kb_passages"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def point_id(passage_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, passage_id))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_passages() -> list[dict]:
    titles = pd.read_csv(PROCESSED / "metadata.csv").set_index("doc_id")["title"].to_dict()
    passages = []
    with open(PROCESSED / "passages.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            p["doc_title"] = titles[p["doc_id"]]
            p["text_hash"] = text_hash(p["text"] + "|" + p["status"] + "|" + p["version"])
            # texte vectorise : titres + contenu (le champ "text" reste la reponse brute)
            p["embed_text"] = f"{p['doc_title']} - {p['section_title']}\n{p['text']}"
            passages.append(p)
    return passages


def index(client: QdrantClient, model: SentenceTransformer) -> None:
    passages = load_passages()
    dim = model.get_sentence_embedding_dimension()

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    # hashes deja presents dans la base
    existing, offset = {}, None
    while True:
        points, offset = client.scroll(COLLECTION, limit=256, offset=offset,
                                       with_payload=["text_hash"], with_vectors=False)
        existing.update({str(p.id): p.payload.get("text_hash") for p in points})
        if offset is None:
            break

    todo = [p for p in passages if existing.get(point_id(p["passage_id"])) != p["text_hash"]]
    print(f"{len(passages)} passages, {len(todo)} a (re)indexer, {len(passages) - len(todo)} inchanges")
    if not todo:
        return

    vectors = model.encode([p["embed_text"] for p in todo], normalize_embeddings=True,
                           show_progress_bar=True)
    client.upsert(COLLECTION, points=[
        models.PointStruct(
            id=point_id(p["passage_id"]),
            vector=v.tolist(),
            payload={k: p[k] for k in ("passage_id", "doc_id", "doc_title", "version", "section",
                                       "section_title", "response_id", "department",
                                       "confidentiality", "allowed_roles", "status",
                                       "text", "text_hash")},
        )
        for p, v in zip(todo, vectors)
    ])
    print(f"{len(todo)} passages indexes. Total dans la base : {client.count(COLLECTION).count}")


def search(client, model, query: str, role: str, k: int = 5):
    """Recherche filtree : seulement les passages en vigueur ET autorises pour ce role."""
    vec = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    flt = models.Filter(must=[
        models.FieldCondition(key="status", match=models.MatchValue(value="en_vigueur")),
        models.FieldCondition(key="allowed_roles", match=models.MatchValue(value=role)),
    ])
    return client.query_points(COLLECTION, query=vec, query_filter=flt, limit=k,
                               with_payload=True).points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="question de test a lancer apres l'indexation")
    parser.add_argument("--role", default="Employee")
    args = parser.parse_args()

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    model = SentenceTransformer(MODEL_NAME)

    index(client, model)

    if args.test:
        print(f"\nQuestion : {args.test} (role : {args.role})")
        for hit in search(client, model, args.test, args.role):
            pl = hit.payload
            print(f"  {hit.score:.3f}  {pl['passage_id']}  [{pl['confidentiality']}]  {pl['section_title']}")


if __name__ == "__main__":
    main()