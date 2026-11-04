"""
Fine-tuning d'un reranker (cross-encodeur) pour le Secure Enterprise Knowledge Assistant.

Pipeline :
1. Retriever de reference : bge-small (meme modele que 5-index.py), sur les 64 passages.
2. Negatifs difficiles : pour chaque question d'entrainement, les passages les mieux classes
   par le retriever qui ne sont pas la bonne reponse.
3. Entrainement d'un cross-encodeur sur des paires (question, passage) avec label 1 (bon passage)
   ou 0 (negatif difficile).
4. Evaluation sur le split TEST uniquement, pour trois configurations :
   retriever seul, reranker pre-entraine, reranker affine.

Regles :
- Le split test n'est jamais utilise pour l'entrainement.
- Les donnees synthetiques (data/synthetic/) ne sont jamais utilisees ici.

Usage (depuis le dossier scripts/) :
    python 7-finetune_reranker.py --base cross-encoder/ms-marco-MiniLM-L-6-v2 --epochs 2 --negs 3 --name minilm_ft

Sorties :
    models/<name>/                     modele affine (ignore par Git)
    data/eval/reranker_<name>.json     metriques des trois configurations
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
EVAL = ROOT / "data" / "eval"
MODELS = ROOT / "models"

EMB_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SEED = 42
TOP_K = 10


def load_data():
    """Charge les passages (avec le titre du document) et les questions avec leur split."""
    passages = pd.read_json(PROC / "passages.jsonl", lines=True)
    passages = passages[passages["status"] == "en_vigueur"].reset_index(drop=True)
    assert passages["response_id"].is_unique, "doublons de response_id parmi les passages en vigueur"
    questions = pd.read_csv(PROC / "questions_split.csv")
    plan = pd.read_csv(PROC / "document_plan.csv")

    titles = dict(zip(plan["doc_id"], plan["doc_title"]))
    passages["doc_title"] = passages["doc_id"].map(titles)
    # Meme forme que embed_text dans 5-index.py : "titre - section" puis retour a la ligne puis texte.
    passages["ptext"] = (
        passages["doc_title"] + " - " + passages["section_title"] + "\n" + passages["text"]
    )
    return passages.reset_index(drop=True), questions


def encode(model, texts):
    return model.encode(list(texts), normalize_embeddings=True, batch_size=32, show_progress_bar=False)


def retrieve(emb_model, p_emb, questions_txt):
    """Top-K passages par similarite cosinus, comme le retriever de reference (sans filtre de role)."""
    q_emb = encode(emb_model, [QUERY_PREFIX + q for q in questions_txt])
    sims = q_emb @ p_emb.T
    return np.argsort(-sims, axis=1)[:, :TOP_K]


def rerank(ce, questions_txt, top, passages):
    """Reordonne les TOP_K candidats de chaque question avec le cross-encodeur."""
    ordered = []
    for q, cand in zip(questions_txt, top):
        pairs = [(q, passages.loc[i, "ptext"]) for i in cand]
        scores = np.asarray(ce.predict(pairs, show_progress_bar=False))
        order = np.argsort(-scores)
        ordered.append([cand[j] for j in order])
    return ordered


def build_train_samples(train_df, passages, emb_model, p_emb, negs):
    """Une paire positive par question, plus `negs` negatifs difficiles tires du top du retriever."""
    pos_idx = {rid: i for i, rid in enumerate(passages["response_id"])}
    top = retrieve(emb_model, p_emb, train_df["question"].tolist())

    samples = []
    for (_, row), ranked in zip(train_df.iterrows(), top):
        pos = pos_idx[row["response_id"]]
        samples.append(InputExample(texts=[row["question"], passages.loc[pos, "ptext"]], label=1.0))
        hard_negs = [i for i in ranked if i != pos][:negs]
        for i in hard_negs:
            samples.append(InputExample(texts=[row["question"], passages.loc[i, "ptext"]], label=0.0))
    return samples


def metrics(ranked_lists, positives):
    """Recall@1,3,5,10 et MRR@10 au niveau passage."""
    ranks = []
    for ranked, pos in zip(ranked_lists, positives):
        ranked = list(ranked)
        ranks.append(ranked.index(pos) + 1 if pos in ranked else None)

    out = {}
    for k in (1, 3, 5, 10):
        out[f"recall@{k}"] = round(float(np.mean([r is not None and r <= k for r in ranks])), 4)
    out["mrr@10"] = round(float(np.mean([1 / r if r is not None and r <= 10 else 0.0 for r in ranks])), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    help="reranker pre-entraine de depart (MiniLM pour CPU, bge-reranker-base plus lourd)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--negs", type=int, default=3, help="negatifs difficiles par question")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--name", default="reranker_ft")
    ap.add_argument("--limit", type=int, default=0,
                    help="n'utiliser que N questions d'entrainement (test rapide, 0 = tout)")
    ap.add_argument("--max-length", type=int, default=256,
                    help="longueur max des paires question+passage (128 pour aller plus vite)")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    passages, questions = load_data()
    emb = SentenceTransformer(EMB_MODEL)
    p_emb = encode(emb, passages["ptext"])
    pos_idx = {rid: i for i, rid in enumerate(passages["response_id"])}

    # --- Split test : utilise uniquement pour l'evaluation ---
    test = questions[questions["split"] == "test"].reset_index(drop=True)
    test_q = test["question"].tolist()
    test_top = retrieve(emb, p_emb, test_q)
    test_pos = [pos_idx[r] for r in test["response_id"]]

    results = {}

    # 1. Retriever seul
    results["retriever_only"] = metrics(test_top, test_pos)
    print("Retriever seul :", results["retriever_only"])
    print("  (doit etre proche de la baseline : Recall@1 = 0,756 ; MRR@10 = 0,864)")

    # 2. Reranker pre-entraine, sans affinage
    base_ce = CrossEncoder(args.base, max_length=args.max_length)
    results["reranker_pretrained"] = metrics(rerank(base_ce, test_q, test_top, passages), test_pos)
    print("Reranker pre-entraine :", results["reranker_pretrained"])

    # 3. Fine-tuning sur le split train uniquement
    train = questions[questions["split"] == "train"].reset_index(drop=True)
    if args.limit:
        train = train.head(args.limit)
        print(f"MODE TEST RAPIDE : {len(train)} questions d'entrainement")
    else:
        assert len(train) == 336, f"Split train inattendu : {len(train)} questions"
    samples = build_train_samples(train, passages, emb, p_emb, args.negs)
    n_pos = sum(s.label == 1.0 for s in samples)
    print(f"Entrainement : {len(samples)} paires ({n_pos} positives, {len(samples) - n_pos} negatives)")

    loader = DataLoader(samples, shuffle=True, batch_size=args.batch)
    ft = CrossEncoder(args.base, num_labels=1, max_length=args.max_length)
    out_dir = MODELS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ft.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(loader) * args.epochs),
        optimizer_params={"lr": args.lr},
        output_path=str(out_dir),
        show_progress_bar=True,
    )
    duration = round(time.time() - t0, 1)

    # 4. Evaluation du modele affine sur le test
    results["reranker_finetuned"] = metrics(rerank(ft, test_q, test_top, passages), test_pos)
    print("Reranker affine :", results["reranker_finetuned"])

    report = {
        "base_model": args.base,
        "epochs": args.epochs,
        "negs_per_question": args.negs,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "seed": SEED,
        "train_questions": len(train),
        "train_pairs": len(samples),
        "train_duration_seconds": duration,
        "test_questions": len(test),
        "results": results,
    }
    EVAL.mkdir(parents=True, exist_ok=True)
    out_json = EVAL / f"reranker_{args.name}.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultats ecrits dans {out_json}")


if __name__ == "__main__":
    main()