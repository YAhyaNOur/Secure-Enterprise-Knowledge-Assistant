"""
Verification des donnees synthetiques contre l'index Qdrant.

1. synthetic/access_tests.csv : pour chaque cas, l'utilisateur (users.csv) ne doit recevoir
   que des passages autorises pour son role (0 fuite), et :
   - si expected = "refus" : le document interdit ne doit pas apparaitre ;
   - sinon : le document attendu doit apparaitre dans les k premiers resultats.

2. synthetic/unanswerable.csv : questions hors corpus. La recherche seule ne peut pas refuser :
   on mesure le score du meilleur passage et on le compare a celui de questions repondables.
   Le refus effectif se verifie a l'etape de generation (RAG).

Usage (depuis scripts/) :
    python check_synthetic.py --k 5

Sorties : data/eval/synthetic_checks.csv (detail), data/eval/synthetic_checks.json (resume)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "synthetic"
PROCESSED = ROOT / "data" / "processed"
EVAL = ROOT / "data" / "eval"
QDRANT_PATH = ROOT / "data" / "qdrant"
COLLECTION = "kb_passages"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def search(client, model, question: str, role: str, k: int):
    vec = model.encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()
    flt = models.Filter(must=[
        models.FieldCondition(key="status", match=models.MatchValue(value="en_vigueur")),
        models.FieldCondition(key="allowed_roles", match=models.MatchValue(value=role)),
    ])
    return client.query_points(COLLECTION, query=vec, query_filter=flt, limit=k,
                               with_payload=["doc_id", "allowed_roles", "version"]).points


def check_access(client, model, users: pd.DataFrame, tests: pd.DataFrame, k: int) -> pd.DataFrame:
    role_of = dict(zip(users["user_id"], users["role"]))
    rows = []
    for _, t in tests.iterrows():
        role = role_of[t["user_id"]]
        hits = search(client, model, t["question"], role, k)
        docs = [h.payload["doc_id"] for h in hits]
        leaks = sum(role not in h.payload["allowed_roles"] for h in hits)
        if t["expected"] == "refus":
            ok = t["expected_doc"] not in docs and leaks == 0
        else:
            ok = t["expected_doc"] in docs and leaks == 0
        rows.append({
            "test_id": t["test_id"], "role": role, "type": t["type"],
            "expected": t["expected"], "returned_docs": " ".join(docs),
            "leaks": leaks, "ok": ok,
        })
    return pd.DataFrame(rows)


def score_of_best(client, model, questions, role: str) -> np.ndarray:
    scores = []
    for q in questions:
        hits = search(client, model, q, role, 1)
        scores.append(hits[0].score if hits else 0.0)
    return np.array(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    EVAL.mkdir(parents=True, exist_ok=True)

    users = pd.read_csv(SYN / "users.csv")
    tests = pd.read_csv(SYN / "access_tests.csv")
    unans = pd.read_csv(SYN / "unanswerable.csv")
    questions = pd.read_csv(PROCESSED / "questions_split.csv")

    client = QdrantClient(path=str(QDRANT_PATH))
    model = SentenceTransformer(MODEL_NAME)

    # 1. Acces
    acc = check_access(client, model, users, tests, args.k)
    summary = {
        "access_tests": len(acc),
        "access_ok": int(acc["ok"].sum()),
        "leaks_total": int(acc["leaks"].sum()),
    }
    for typ in ("interdite", "accessible"):
        sub = acc[acc["type"] == typ]
        if len(sub):
            summary[f"{typ}_ok"] = f"{int(sub['ok'].sum())}/{len(sub)}"

    # 2. Questions sans reponse : score du meilleur passage vs questions repondables
    answerable = questions[questions["split"] == "test"].sample(n=50, random_state=42)["question"]
    s_ans = score_of_best(client, model, answerable.tolist(), "Administrator")
    s_un = score_of_best(client, model, unans["question"].tolist(), "Administrator")
    seuil = float(np.percentile(s_ans, 5))  # 95 % des questions repondables passent
    summary["unanswerable_scores"] = {
        "repondables_median": round(float(np.median(s_ans)), 4),
        "repondables_p5": round(seuil, 4),
        "sans_reponse_median": round(float(np.median(s_un)), 4),
        "sans_reponse_max": round(float(np.max(s_un)), 4),
        "sans_reponse_sous_seuil": f"{int((s_un < seuil).sum())}/{len(s_un)}",
    }
    summary["note"] = ("La recherche seule ne refuse pas : le seuil indique si un refus automatique "
                       "est envisageable. Le refus reel se verifie a la generation.")

    acc.to_csv(EVAL / "synthetic_checks.csv", index=False)
    (EVAL / "synthetic_checks.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["leaks_total"] > 0 or summary["access_ok"] != summary["access_tests"]:
        print("\nATTENTION : voir data/eval/synthetic_checks.csv (lignes ok = False)")


if __name__ == "__main__":
    main()
