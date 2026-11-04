"""
Etape 4b : generer la version V2 des documents modifies (DONNEES SYNTHETIQUES).

Entrees : synthetic/version_changes.csv (source saisie a la main)
          data/processed/passages.jsonl, metadata.csv, data/corpus/<doc>_V1.md (produits par 4)
Sorties : data/corpus/<doc>_V2.md
          data/processed/passages.jsonl : V1 des documents modifies passees en "archive", V2 ajoutees
          data/processed/metadata.csv   : version et date mises a jour pour ces documents

A lancer APRES 4-generate_corpus.py et AVANT 5-index.py.
Idempotent : relancer 4 puis 4b reconstruit l'etat attendu.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "synthetic"
PROCESSED = ROOT / "data" / "processed"
CORPUS = ROOT / "data" / "corpus"
PASSAGES = PROCESSED / "passages.jsonl"
METADATA = PROCESSED / "metadata.csv"
UPDATED_V2 = "2026-01-15"  # date fictive de la V2


def write_v2_markdown(doc_id: str, passages: list[dict]) -> Path:
    titles = pd.read_csv(METADATA).set_index("doc_id")["title"].to_dict()
    first = passages[0]
    title, dept = titles[doc_id], first["department"]
    conf, roles = first["confidentiality"], first["allowed_roles"]
    lines = [
        "---",
        f"doc_id: {doc_id}",
        f"title: {title}",
        f"department: {dept}",
        "version: V2",
        "status: en_vigueur",
        f"confidentiality: {conf}",
        f"allowed_roles: [{', '.join(roles)}]",
        "created: 2025-01-01",
        f"updated: {UPDATED_V2}",
        "synthetic: true",
        "---",
        "",
        f"# {title}",
        "",
        f"NovaTech Solutions | {dept} | Version V2",
        "",
    ]
    for p in sorted(passages, key=lambda x: x["section"]):
        lines += [f"## {p['section']}. {p['section_title']}", "", p["text"], ""]
    path = CORPUS / f"{doc_id}_V2.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    changes = pd.read_csv(SYN / "version_changes.csv")
    all_passages = [json.loads(line) for line in PASSAGES.open(encoding="utf-8")]

    # Repart d'un etat propre : on retire les V2 eventuelles, on remet les V1 en vigueur
    v1 = [p for p in all_passages if p["version"] == "V1"]
    for p in v1:
        p["status"] = "en_vigueur"

    new_v2 = []
    meta = pd.read_csv(METADATA)
    for doc_id in sorted(changes["doc_id"].unique()):
        doc_v1 = [p for p in v1 if p["doc_id"] == doc_id]
        assert doc_v1, f"document inconnu : {doc_id}"
        doc_changes = changes[changes["doc_id"] == doc_id]

        # Chaque reponse modifiee doit appartenir au document
        known = {p["response_id"] for p in doc_v1}
        unknown = set(doc_changes["response_id"]) - known
        assert not unknown, f"{doc_id} : reponses inconnues {sorted(unknown)}"

        new_text = dict(zip(doc_changes["response_id"], doc_changes["v2_text"]))
        doc_v2 = []
        for p in doc_v1:
            q = dict(p)
            q["version"] = "V2"
            q["passage_id"] = f"{doc_id}-V2-S{p['section']:02d}"
            q["text"] = new_text.get(p["response_id"], p["text"])
            q["status"] = "en_vigueur"
            doc_v2.append(q)
            p["status"] = "archive"  # la V1 de ce document n'est plus en vigueur

        new_v2 += doc_v2
        write_v2_markdown(doc_id, doc_v2)

        # Passer la V1 md en archive
        v1_path = CORPUS / f"{doc_id}_V1.md"
        v1_text = v1_path.read_text(encoding="utf-8")
        v1_path.write_text(v1_text.replace("status: en_vigueur", "status: archive", 1), encoding="utf-8")

        # Metadonnees : le document pointe maintenant sur la V2
        mask = meta["doc_id"] == doc_id
        meta.loc[mask, "version"] = "V2"
        meta.loc[mask, "date_maj"] = UPDATED_V2
        meta.loc[mask, "chemin_fichier"] = (CORPUS / f"{doc_id}_V2.md").relative_to(ROOT).as_posix()

    with open(PASSAGES, "w", encoding="utf-8") as f:
        for p in all_passages:
            if p["version"] == "V1":
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        for p in new_v2:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta.to_csv(METADATA, index=False)

    print(f"{len(changes)} reponses modifiees dans {changes['doc_id'].nunique()} documents")
    print(f"V2 creees : {len(new_v2)} passages. V1 archivees : {sum(p['status'] == 'archive' for p in v1)}")


if __name__ == "__main__":
    main()
