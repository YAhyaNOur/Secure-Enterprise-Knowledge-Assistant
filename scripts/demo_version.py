"""
Demonstration de la gestion des versions (DONNEES SYNTHETIQUES pour les V2).

Pour chaque question de synthetic/version_tests.csv, la recherche filtree (status en_vigueur)
doit :
- renvoyer en premier un passage V2 ;
- ne renvoyer AUCUNE V1 du document modifie (la V1 est archivee).
Les V1 des autres documents, non modifies, sont legitimes et ne sont pas comptees.

Usage (depuis scripts/), apres 4-generate_corpus.py, 4b-generate_v2.py et 5-index.py :
    python demo_version.py --role Employee
"""
import argparse
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "synthetic"
QDRANT_PATH = ROOT / "data" / "qdrant"
COLLECTION = "kb_passages"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="Employee")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    tests = pd.read_csv(SYN / "version_tests.csv")
    changes = pd.read_csv(SYN / "version_changes.csv")
    changed_docs = set(changes["doc_id"])

    client = QdrantClient(path=str(QDRANT_PATH))
    model = SentenceTransformer(MODEL_NAME)

    flt = models.Filter(must=[
        models.FieldCondition(key="status", match=models.MatchValue(value="en_vigueur")),
        models.FieldCondition(key="allowed_roles", match=models.MatchValue(value=args.role)),
    ])

    rows = []
    for _, t in tests.iterrows():
        vec = model.encode(QUERY_PREFIX + t["question"], normalize_embeddings=True).tolist()
        hits = client.query_points(COLLECTION, query=vec, query_filter=flt, limit=args.k,
                                   with_payload=["passage_id", "doc_id", "version", "status",
                                                 "response_id"]).points
        top = hits[0].payload if hits else {}
        # V1 d'un document modifie : ne doit jamais sortir
        v1_archivee = [h.payload["passage_id"] for h in hits
                       if h.payload["version"] == "V1" and h.payload["doc_id"] in changed_docs]
        # Tout passage renvoye doit etre en vigueur (filtre) : verification de securite
        hors_statut = [h.payload["passage_id"] for h in hits if h.payload["status"] != "en_vigueur"]
        rows.append({
            "test": t["version_test_id"],
            "doc": t["doc_id"],
            "top1": top.get("passage_id"),
            "top1_version": top.get("version"),
            "bonne_reponse_en_1er": top.get("response_id") == t["response_id"],
            "V1_archivee_renvoyee": len(v1_archivee),
            "hors_statut": len(hors_statut),
        })

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))

    n = len(res)
    print("\n--- Bilan ---")
    print(f"Questions : {n}")
    print(f"Top 1 en V2 : {(res['top1_version'] == 'V2').sum()} / {n}")
    print(f"Bonne reponse en 1er : {res['bonne_reponse_en_1er'].sum()} / {n}")
    print(f"V1 archivee renvoyee (doit etre 0) : {(res['V1_archivee_renvoyee'] > 0).sum()}")
    print(f"Passages hors statut en vigueur (doit etre 0) : {res['hors_statut'].sum()}")


if __name__ == "__main__":
    main()