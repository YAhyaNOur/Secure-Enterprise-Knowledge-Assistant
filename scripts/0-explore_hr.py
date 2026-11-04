from datasets import load_dataset
import pandas as pd

#  dataset
dataset = load_dataset(
    "adarsh-08/hr-policy-dataset",
    split="train"
)

#  DataFrame
df = dataset.to_pandas()

# les premières lignes
print(df.head(10))

#  les dimensions
print("Dimensions :", df.shape)

#  Compter les questions
print("Nombre de questions :", df["instruction"].nunique())

#  les réponses uniques
print("Nombre de réponses uniques :", df["response"].nunique())

# Compter les questions associées à chaque réponse
questions_per_response = df.groupby("response")["instruction"].agg(
    ["count", "nunique"]
)

print(questions_per_response)

# chaque reponse posséde combien de questions
print("\nRépartition du nombre de questions par réponse :")

print(df.groupby("response")["instruction"].nunique().value_counts())


for response, group in df.groupby("response"):

    print("\n" + "=" * 80)
    print("RÉPONSE :", response)

    for question in group["instruction"]:
        print("-", question)