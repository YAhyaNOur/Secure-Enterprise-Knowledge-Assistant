
"""
Etape 1 : Telecharger et verifier le dataset depuis Hugging Face.


Sortie : data/raw/hr_dataset_full.csv
Colonnes : instruction, response
"""

import hashlib
from pathlib import Path

from datasets import load_dataset



# 1. CONFIGURATION


DATASET_ID = "adarsh-08/hr-policy-dataset"
# Valeurs observees lors de l'exploration (voir 0-explore_hr.py)
EXPECTED_ROWS = 512
EXPECTED_QUESTIONS = 512
EXPECTED_ANSWERS = 64
QUESTIONS_PER_ANSWER = 8

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "hr_dataset_full.csv"


def main() -> None:

    
    # 2. TELECHARGEMENT DU DATASET
    

    dataset = load_dataset(DATASET_ID, split="train")

    df = dataset.to_pandas()[["instruction", "response"]]

    print("Dataset telecharge depuis Hugging Face.")

    
    # 3. VERIFICATION DE LA STRUCTURE
  

    n_rows = len(df)
    n_questions = df["instruction"].nunique()
    n_answers = df["response"].nunique()

    print("\n--- Verification generale ---")
    print(f"Nombre de lignes : {n_rows}")
    print(f"Questions uniques : {n_questions}")
    print(f"Reponses uniques : {n_answers}")

    
    questions_per_response = (
        df.groupby("response")["instruction"].nunique()
    )

    print("\n--- Questions par reponse ---")
    print(questions_per_response.value_counts().sort_index())

    # Vérifier les doublons de questions
    duplicate_questions = df["instruction"].duplicated().sum()

    print(f"\nQuestions dupliquees : {duplicate_questions}")

    
    # 4. VALIDATION DES RESULTATS ATTENDUS
   

    if (
        n_rows != EXPECTED_ROWS
        or n_questions != EXPECTED_QUESTIONS
        or n_answers != EXPECTED_ANSWERS
        or duplicate_questions != 0
        or not (questions_per_response == QUESTIONS_PER_ANSWER).all()
    ):
        raise SystemExit(
            "ERREUR : le dataset ne correspond pas "
            "aux caracteristiques attendues. "
            "Verifiez-le avant de continuer."
        )

    print("\nToutes les verifications sont reussies !")

   
    # 5. EXPORT CSV


    OUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT, index=False, encoding="utf-8")

    print(f"\nFichier sauvegarde : {OUT}")

   
    # 6. EMPREINTE SHA256

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()[:12]

    print(f"Empreinte SHA256 : {digest}")
    print("Note cette empreinte dans ton rapport.")


if __name__ == "__main__":
    main()
