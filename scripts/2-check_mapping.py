
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "hr_dataset_full.csv"

df = pd.read_csv(INPUT)

# Reproduire exactement l'attribution des IDs de prepare_data.py
answers = (
    df.drop_duplicates("response")[["response"]]
    .reset_index(drop=True)
)

answers["response_id"] = [
    f"R{i + 1:02d}" for i in answers.index
]

# Afficher chaque réponse avec ses 8 questions
for _, row in answers.iterrows():

    rid = row["response_id"]
    response = row["response"]

    questions = df[df["response"] == response]["instruction"]

    print("\n" + "=" * 90)
    print(f"{rid} | Nombre de questions : {len(questions)}")
    print("\nREPONSE :")
    print(response)

    print("\nQUESTIONS :")
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
