"""
Preparation  des donnees :

Entree  : data/raw/hr_dataset_full.csv  (dataset adarsh-08/hr-policy-dataset, colonnes instruction, response)
Sorties : data/processed/document_plan.csv, answers_by_document.csv, questions_split.csv
"""
import random
from pathlib import Path

import pandas as pd

SEED = 42
TEST_RATIO = 0.30
ROOT = Path(__file__).resolve().parents[1]  # racine du projet (dossier parent de scripts/)
INPUT = ROOT / "data" / "raw" / "hr_dataset_full.csv"
OUT = ROOT / "data" / "processed"

ALL_ROLES = "Employee;HR Manager;IT Administrator;Finance Manager;Administrator"
HR_ONLY = "HR Manager;Administrator"
HR_FIN = "HR Manager;Finance Manager;Administrator"


def rng(lo: int, hi: int) -> list[str]:
    return [f"R{i:02d}" for i in range(lo, hi + 1)]


# (doc_id, titre, departement, confidentialite, roles autorises, reponses du dataset)
DOCUMENTS = [
    ("HR-01", "Working hours, attendance and overtime", "HR", "public_interne", ALL_ROLES, ["R01", "R02", "R22"]),
    ("HR-02", "Remote and hybrid work policy", "HR", "public_interne", ALL_ROLES, ["R03", "R04"]),
    ("HR-03", "Workplace conduct and dress code", "HR", "public_interne", ALL_ROLES, ["R05", "R06", "R07", "R57"]),
    ("HR-04", "Leave policy", "HR", "public_interne", ALL_ROLES, rng(8, 15) + ["R64"]),
    ("HR-05", "Onboarding and probation", "HR", "public_interne", ALL_ROLES, ["R31", "R33", "R34"]),
    ("HR-06", "Recruitment and hiring process", "HR", "restreint", HR_ONLY, ["R29", "R30", "R32"]),
    ("HR-07", "Performance reviews and promotions", "HR", "public_interne", ALL_ROLES, ["R35", "R36", "R37"]),
    ("HR-08", "Salary revision and performance improvement plans", "HR", "confidentiel", HR_ONLY, ["R38", "R39"]),
    ("HR-09", "Training and career development", "HR", "public_interne", ALL_ROLES, rng(40, 43)),
    ("HR-10", "Health benefits and wellness", "HR", "public_interne", ALL_ROLES, ["R23", "R24", "R25", "R27", "R28"]),
    ("HR-11", "Employee relations, grievances and ethics", "HR", "public_interne", ALL_ROLES, rng(53, 56)),
    ("HR-12", "Resignation, offboarding and final settlement", "HR", "public_interne", ALL_ROLES, rng(58, 63)),
    ("FIN-01", "Payroll, tax and provident fund", "Finance", "public_interne", ALL_ROLES, ["R16", "R17", "R18", "R26"]),
    ("FIN-02", "Bonus and incentive programs", "Finance", "restreint", HR_FIN, ["R19", "R20"]),
    ("FIN-03", "Expense reimbursement and business travel", "Finance", "public_interne", ALL_ROLES, ["R21"] + rng(44, 47)),
    ("IT-01", "IT security and acceptable use", "IT", "public_interne", ALL_ROLES, ["R48", "R49", "R52"]),
    ("POL-01", "Data protection and confidentiality", "Company Policies", "public_interne", ALL_ROLES, ["R50", "R51"]),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)[["instruction", "response"]]

    # 1. Reponses uniques, identifiees dans l'ordre d'apparition (le dataset est trie par sujet)
    answers = df.drop_duplicates("response")[["response"]].reset_index(drop=True)
    answers["response_id"] = [f"R{i + 1:02d}" for i in answers.index]
    answers["n_questions"] = answers["response"].map(df.groupby("response").size())
    assert len(df) == 512 and len(answers) == 64 and (answers["n_questions"] == 8).all()

    # 2. Rattachement de chaque reponse a un document
    rows = []
    for doc_id, title, dept, conf, roles, rids in DOCUMENTS:
        for section, rid in enumerate(rids, start=1):
            rows.append(dict(response_id=rid, doc_id=doc_id, doc_title=title, section=section,
                             department=dept, confidentiality=conf, allowed_roles=roles))
    mapping = pd.DataFrame(rows)
    assert mapping["response_id"].is_unique
    assert set(mapping["response_id"]) == set(answers["response_id"]), "reponse oubliee ou inconnue"

    # 3. Decoupage train/test par reponse, equilibre par document
    random.seed(SEED)
    mapping["split"] = "train"
    for _, grp in mapping.groupby("doc_id"):
        n = len(grp)
        n_test = 1 if n == 2 else max(1, round(TEST_RATIO * n))
        for idx in random.sample(list(grp.index), n_test):
            mapping.loc[idx, "split"] = "test"

    # 4. Fichiers de sortie
    by_doc = mapping.merge(answers, on="response_id")[
        ["response_id", "doc_id", "doc_title", "section", "department", "confidentiality",
         "allowed_roles", "split", "n_questions", "response"]
    ].sort_values(["doc_id", "section"])
    by_doc.to_csv(OUT / "answers_by_document.csv", index=False)

    plan = mapping.groupby(["doc_id", "doc_title", "department", "confidentiality", "allowed_roles"]).agg(
        reponses=("response_id", lambda s: ", ".join(s)),
        n=("response_id", "size"),
        test=("split", lambda s: int((s == "test").sum())),
    ).reset_index()
    plan.to_csv(OUT / "document_plan.csv", index=False)

    questions = df.merge(answers[["response", "response_id"]], on="response").merge(
        mapping.drop(columns=["section"]), on="response_id")
    questions = questions.rename(columns={"instruction": "question"})
    questions.insert(0, "question_id", [f"Q{i + 1:03d}" for i in range(len(questions))])
    questions[["question_id", "question", "response_id", "doc_id", "doc_title", "department",
               "confidentiality", "allowed_roles", "split"]].to_csv(OUT / "questions_split.csv", index=False)

    print(f"{len(answers)} reponses -> {len(plan)} documents")
    print("Questions par ensemble :", questions["split"].value_counts().to_dict())
    print(f"Fichiers ecrits dans {OUT.resolve()}")


if __name__ == "__main__":
    main()