"""
Evaluation de la baseline de retrieval : Recall@k, MRR et test de fuite par role.

Entrees : data/processed/questions_split.csv, base Qdrant (kb_passages)
Sorties : data/eval/baseline_<split>_<role>.csv  (une ligne par question)
          data/eval/baseline_<split>_<role>.json (resume, dont le nombre de fuites)

Usage (depuis scripts/) :
    python 6-eval_baseline.py --split test --role Employee
    python 6-eval_baseline.py --split test --role Employee --no-filter   (controle : filtre de droits desactive)

Fuite : passage renvoye dont le role demande n'est pas dans allowed_roles. Objectif : 0.
Le controle --no-filter doit en produire, sinon le compteur ne prouve rien.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EVAL = ROOT / "data" / "eval"
QDRANT_PATH = ROOT / "data" / "qdrant"
COLLECTION = "kb_passages"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
K_MAX = 10
KS = (1, 3, 5, 10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--role", default="Administrator")
    parser.add_argument("--no-filter", action="store_true",
                        help="controle : ne pas filtrer par role (doit produire des fuites)")
    args = parser.parse_args()

    EVAL.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED / "questions_split.csv")
    df = df[df["split"] == args.split].reset_index(drop=True)
    mode = "SANS FILTRE ROLE (controle)" if args.no_filter else "avec filtre role"
    print(f"{len(df)} questions ({args.split}), role : {args.role}, {mode}")

    client = QdrantClient(path=str(QDRANT_PATH))
    model = SentenceTransformer(MODEL_NAME)
    qvecs = model.encode([QUERY_PREFIX + q for q in df["question"]],
                         normalize_embeddings=True, batch_size=32)

    must = [models.FieldCondition(key="status", match=models.MatchValue(value="en_vigueur"))]
    if not args.no_filter:
        must.append(models.FieldCondition(key="allowed_roles", match=models.MatchValue(value=args.role)))
    flt = models.Filter(must=must)

    rows, n_fuites = [], 0
    for (_, r), vec in zip(df.iterrows(), qvecs):
        hits = client.query_points(COLLECTION, query=vec.tolist(), query_filter=flt, limit=K_MAX,
                                   with_payload=["response_id", "doc_id", "allowed_roles"]).points
        rids = [h.payload["response_id"] for h in hits]
        dids = [h.payload["doc_id"] for h in hits]

        # Fuites : passages renvoyes dont le role demande n'est pas autorise
        fuites = [h.payload["doc_id"] for h in hits if args.role not in h.payload["allowed_roles"]]
        n_fuites += len(fuites)

        rows.append({
            "question_id": r["question_id"], "question": r["question"],
            "gold_response": r["response_id"], "gold_doc": r["doc_id"],
            "accessible": args.role in r["allowed_roles"].split(";"),
            "rank_passage": rids.index(r["response_id"]) + 1 if r["response_id"] in rids else None,
            "rank_doc": dids.index(r["doc_id"]) + 1 if r["doc_id"] in dids else None,
            "top1": rids[0] if rids else None,
            "fuites": len(fuites),
        })

    res = pd.DataFrame(rows)
    for col in ("rank_passage", "rank_doc"):
        res[col] = pd.to_numeric(res[col])

    summary = {
        "split": args.split, "role": args.role, "filtre_role": not args.no_filter,
        "n_questions": len(res), "model": MODEL_NAME,
        "fuites": n_fuites,
    }
    for level, col in (("passage", "rank_passage"), ("document", "rank_doc")):
        for k in KS:
            summary[f"recall@{k}_{level}"] = round(float((res[col] <= k).mean()), 4)
        summary[f"mrr@{K_MAX}_{level}"] = round(float((1 / res[col]).fillna(0).mean()), 4)

    # Metriques sur les seules questions accessibles au role
    acc = res[res["accessible"]]
    if len(acc):
        summary["recall@1_passage_accessibles"] = round(float((acc["rank_passage"] <= 1).mean()), 4)
        summary[f"mrr@{K_MAX}_passage_accessibles"] = round(float((1 / acc["rank_passage"]).fillna(0).mean()), 4)
    summary["n_accessibles"] = int(res["accessible"].sum())

    suffix = f"{args.split}_{args.role.replace(' ', '_')}" + ("_nofilter" if args.no_filter else "")
    res.to_csv(EVAL / f"baseline_{suffix}.csv", index=False)
    (EVAL / f"baseline_{suffix}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")

    print("\n--- Resultats ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    misses = res[~(res["rank_passage"] <= 1)].sort_values("rank_passage", na_position="last")
    print(f"\n--- Questions dont le bon passage n'est pas en 1er ({len(misses)}) ---")
    for _, m in misses.head(15).iterrows():
        print(f"[{m['gold_response']} attendu, {m['top1']} en 1er, rang={m['rank_passage']}] {m['question']}")


if __name__ == "__main__":
    main()