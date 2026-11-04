"""
Entree  : data/processed/answers_by_document.csv  
Sorties : data/corpus/<doc_id>_V1.md               
          data/processed/metadata.csv              
          data/processed/passages.jsonl            

"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # racine du projet (dossier parent de scripts/)
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
CORPUS = DATA / "corpus"
VERSION = "V1"
CREATED = "2025-01-01"  # date fictive de creation de la V1

# Titre de section pour chaque reponse du dataset (choix du projet).
SECTION_TITLES = {
    "R01": "Standard working hours", "R02": "Attendance tracking", "R22": "Overtime",
    "R03": "Remote work", "R04": "Hybrid work model",
    "R05": "Office conduct", "R06": "Dress code", "R07": "Employee responsibilities", "R57": "Code of Conduct",
    "R08": "Casual leave", "R09": "Sick leave", "R10": "Earned leave", "R11": "Maternity leave",
    "R12": "Paternity leave", "R13": "Bereavement leave", "R14": "Unpaid leave",
    "R15": "Leave requests and approval", "R64": "Summary of paid leave",
    "R31": "First day and joining formalities", "R33": "Probation period", "R34": "Orientation program",
    "R29": "Recruitment process", "R30": "Offer letter", "R32": "Background verification",
    "R35": "Performance reviews", "R36": "Goal setting", "R37": "Promotions",
    "R38": "Salary revisions", "R39": "Performance Improvement Plan",
    "R40": "Training programs", "R41": "Professional certifications",
    "R42": "Learning Management System", "R43": "Individual Development Plans",
    "R23": "Group health insurance", "R24": "Dental coverage", "R25": "Vision benefits",
    "R27": "Wellness program", "R28": "Employee Assistance Program",
    "R53": "Grievance redressal", "R54": "Anti-harassment policy", "R55": "Conflict resolution",
    "R56": "Ethics hotline",
    "R58": "Resignation procedure", "R59": "Notice period", "R60": "Exit interview",
    "R61": "Final settlement", "R62": "Experience letter", "R63": "Relieving letter",
    "R16": "Salary cycle", "R17": "Payslips", "R18": "Tax deducted at source",
    "R26": "Provident Fund and pension",
    "R19": "Annual performance bonus", "R20": "Incentive programs",
    "R21": "Expense reimbursement", "R44": "Travel booking", "R45": "Travel expense claims",
    "R46": "Claim processing and status", "R47": "Travel approval workflow",
    "R48": "Password policy", "R49": "Company devices and acceptable use",
    "R52": "Cybersecurity awareness training",
    "R50": "Data security", "R51": "Non-Disclosure Agreement",
}


def main() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED / "answers_by_document.csv")

    missing = set(df["response_id"]) - set(SECTION_TITLES)
    assert not missing, f"titres manquants : {sorted(missing)}"

    meta_rows, passages = [], []
    for doc_id, grp in df.groupby("doc_id", sort=True):
        grp = grp.sort_values("section")
        first = grp.iloc[0]
        roles = first["allowed_roles"].split(";")
        path = CORPUS / f"{doc_id}_{VERSION}.md"

        lines = [
            "---",
            f"doc_id: {doc_id}",
            f"title: {first['doc_title']}",
            f"department: {first['department']}",
            f"version: {VERSION}",
            "status: en_vigueur",
            f"confidentiality: {first['confidentiality']}",
            f"allowed_roles: [{', '.join(roles)}]",
            f"created: {CREATED}",
            f"updated: {CREATED}",
            "---",
            "",
            f"# {first['doc_title']}",
            "",
            f"NovaTech Solutions | {first['department']} | Version {VERSION}",
            "",
        ]
        for _, r in grp.iterrows():
            title = SECTION_TITLES[r["response_id"]]
            lines += [f"## {int(r['section'])}. {title}", "", r["response"].strip(), ""]
            passages.append({
                "passage_id": f"{doc_id}-{VERSION}-S{int(r['section']):02d}",
                "doc_id": doc_id, "version": VERSION, "section": int(r["section"]),
                "section_title": title, "response_id": r["response_id"],
                "department": r["department"], "confidentiality": r["confidentiality"],
                "allowed_roles": roles, "status": "en_vigueur",
                "text": r["response"].strip(),
            })
        path.write_text("\n".join(lines), encoding="utf-8")

        meta_rows.append({
            "doc_id": doc_id, "title": first["doc_title"], "department": first["department"],
            "type": "policy", "version": VERSION, "date_creation": CREATED, "date_maj": CREATED,
            "statut": "en_vigueur", "confidentialite": first["confidentiality"],
            "roles_autorises": first["allowed_roles"], "chemin_fichier": path.relative_to(ROOT).as_posix(),
            "n_sections": len(grp),
        })

    pd.DataFrame(meta_rows).to_csv(PROCESSED / "metadata.csv", index=False)
    with open(PROCESSED / "passages.jsonl", "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"{len(meta_rows)} documents, {len(passages)} passages")
    print(f"Documents dans {CORPUS.resolve()}")


if __name__ == "__main__":
    main()