# Secure Enterprise Knowledge Assistant

Assistant de recherche documentaire pour entreprise, avec **contrôle d'accès par rôle**. Les employés de NovaTech Solutions (entreprise fictive) posent des questions en langage naturel sur les documents internes (RH, finance, IT, politiques) et ne reçoivent que les informations auxquelles leur rôle donne accès.

Projet réalisé par Nour (TEK-UP University, Data Science & AI Engineering), sujet n°5 : *Secure Enterprise Knowledge Assistant*.

## Présentation

Le projet combine une recherche vectorielle filtrée par permissions, un reranker affiné sur des exemples de pertinence, un modèle de génération (Gemini) qui rédige la réponse à partir des passages retrouvés, et un serveur MCP qui expose la recherche et la consultation des permissions à un client externe. Toutes les données sont synthétiques.

Le filtrage des droits est appliqué **dans la requête vectorielle, avant le classement** : un passage interdit n'est jamais récupéré puis masqué, il reste invisible. Seules les versions de documents au statut `en_vigueur` sont retournées.

## Fonctionnement

```mermaid
flowchart LR
    A[Question + utilisateur] --> B[Rôle de l'utilisateur]
    B --> C[Recherche vectorielle Qdrant<br/>filtre rôle + statut]
    C --> D[Top 10 passages]
    D --> E[Reranker]
    E --> F[Top 3-5 passages]
    F --> G[Gemini]
    G --> H[Réponse avec sources<br/>ou refus]
```

Les étapes Reranker et Gemini sont en cours de développement (voir « État d'avancement »).

## Structure du projet

```
ai_assistant/
├── data/
│   ├── raw/             dataset brut téléchargé (hr_dataset_full.csv)
│   ├── processed/       plan des documents, split, metadata.csv, passages.jsonl
│   ├── corpus/          17 documents Markdown (HR-01_V1.md ... POL-01_V1.md)
│   ├── qdrant/          base vectorielle locale (régénérable, ignorée par Git)
│   └── eval/            résultats d'évaluation (baseline_test.csv / .json)
├── scripts/
│   ├── 1-download_hr.py
│   ├── 2-check_mapping.py
│   ├── 3-prepare.py
│   ├── 4-generate_corpus.py
│   ├── 5-index.py
│   └── 6-eval_baseline.py
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

## Installation

Prérequis : Python 3.10 ou plus. Les commandes ci-dessous sont pour Windows PowerShell.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Pour l'étape de génération avec Gemini, copier `.env.example` vers `.env` et renseigner la clé :

```
GEMINI_API_KEY=votre_cle
```

Le fichier `.env` ne doit jamais être versionné.

## Utilisation

Les scripts se lancent dans l'ordre, depuis le dossier `scripts/`, car chacun lit les fichiers produits par le précédent.

```powershell
cd scripts
python 1-download_hr.py
python 3-prepare.py
python 4-generate_corpus.py
python 5-index.py
python 6-eval_baseline.py
```

Le détail de chaque script (ce qu'il fait, pourquoi il existe, ce qu'il lit et produit) est donné dans la section « Description des fichiers » plus bas.

Pour tester la recherche avec contrôle d'accès après l'indexation :

```powershell
python 5-index.py --test "How are PIP goals defined?" --role Employee
python 5-index.py --test "How are PIP goals defined?" --role "HR Manager"
```

Le premier appel ne doit renvoyer aucun passage du document HR-08 (réservé aux RH), le second doit faire remonter le passage sur les plans d'amélioration de la performance. Relancer `5-index.py` sans modification doit afficher « 0 à (re)indexer » : seuls les passages nouveaux ou modifiés sont réindexés, grâce à une empreinte de leur texte.

## Description des fichiers

Cette section explique, pour chaque fichier de code, ce qu'il fait, pourquoi il existe et ce qu'il lit et produit.

### `scripts/1-download_hr.py`

**Ce qu'il fait.** Il télécharge le dataset `adarsh-08/hr-policy-dataset` depuis Hugging Face, garde les colonnes `instruction` (la question) et `response` (la réponse), puis vérifie sa structure : 512 lignes, 512 questions uniques, 64 réponses uniques, exactement 8 questions par réponse et aucune question dupliquée. Si une de ces conditions n'est pas remplie, le script s'arrête avec une erreur. Sinon il exporte le tout en CSV UTF-8 et affiche une empreinte SHA256 du fichier.

**Pourquoi.** Le reste du projet repose sur ces chiffres (par exemple 64 réponses et 8 questions par réponse). Le contrôle garantit qu'on part de données connues et que le pipeline s'arrête si le dataset change. L'empreinte permet de citer dans le rapport la version exacte des données utilisées.

**Entrée / sortie.** Entrée : Hugging Face. Sortie : `data/raw/hr_dataset_full.csv`.

### `scripts/2-check_mapping.py`

**Ce qu'il fait.** Il relit le CSV, attribue à chaque réponse unique un identifiant (R01 à R64, dans l'ordre d'apparition), exactement comme le fait `3-prepare.py`, puis affiche chaque réponse avec ses 8 questions. Il n'écrit aucun fichier : il affiche seulement dans le terminal.

**Pourquoi.** Avant de regrouper les réponses en documents, il fallait lire le contenu et comprendre à quoi correspond chaque identifiant. Ce script a permis de repérer des questions qui ne sont pas couvertes par leur réponse, des réponses qui se recoupent (le total des congés payés avec les trois types de congés, les trois réponses sur les frais de voyage) et le caractère corrompu de R03 lors de l'affichage dans PowerShell. Il sert aussi de dictionnaire pour savoir ce qu'est « R39 » ou « R64 » quand on lit les résultats d'évaluation.

**Entrée / sortie.** Entrée : `data/raw/hr_dataset_full.csv`. Sortie : affichage terminal (pour le sauvegarder proprement, écrire dans un fichier ouvert en UTF-8 plutôt que rediriger la sortie PowerShell).

### `scripts/3-prepare.py`

**Ce qu'il fait.** Il contient la table qui regroupe les 64 réponses en 17 documents (HR-01 à HR-12, FIN-01 à FIN-03, IT-01, POL-01) avec, pour chacun, un titre, un département, un niveau de confidentialité et la liste des rôles autorisés. Il vérifie que chaque réponse appartient à un seul document et qu'aucune n'est oubliée. Il découpe ensuite les données en entraînement et test, par réponse, avec la graine 42 et environ 30 % de test par document (au moins une réponse de test par document, et une sur deux pour les documents de deux réponses). Il écrit enfin trois fichiers de sortie.

**Pourquoi.** Le dataset ne contient ni documents, ni droits d'accès. Pour simuler une vraie base documentaire avec des rôles, ces regroupements et ces permissions sont des choix du projet. Regrouper les réponses voisines dans un même document réduit les confusions entre documents différents. Le découpage par réponse (et non par question) évite qu'une même réponse apparaisse en entraînement et en test, ce qui fausserait l'évaluation du reranker.

**Entrée / sortie.** Entrée : `data/raw/hr_dataset_full.csv`. Sorties dans `data/processed/` : `answers_by_document.csv` (chaque réponse avec son document, sa section et son split), `document_plan.csv` (un résumé par document) et `questions_split.csv` (les 512 questions avec leur identifiant Q001 à Q512, leur réponse, leur document, leurs droits et leur split).

### `scripts/4-generate_corpus.py`

**Ce qu'il fait.** Il lit `answers_by_document.csv` et écrit un fichier Markdown par document. Chaque fichier commence par un en-tête de métadonnées (identifiant, titre, département, version V1, statut `en_vigueur`, confidentialité, rôles autorisés, dates), puis contient une section numérotée par réponse, avec un titre de section choisi par le projet et le texte de la réponse recopié sans modification. Il produit aussi un fichier de métadonnées avec une ligne par document et un fichier de passages avec une ligne par section.

**Pourquoi.** Il transforme des paires question-réponse en documents d'entreprise lisibles et versionnés, ce qui est le point de départ attendu par le sujet (documents avec métadonnées et versions). Le fichier de passages définit l'unité de recherche : une réponse est un passage, avec un identifiant de la forme `HR-04-V1-S03`. Ce même identifiant sert ensuite à l'indexation, à l'évaluation et aux futures mises à jour de version.

**Entrée / sortie.** Entrée : `data/processed/answers_by_document.csv`. Sorties : `data/corpus/<doc_id>_V1.md` (17 fichiers), `data/processed/metadata.csv` et `data/processed/passages.jsonl` (64 passages).

### `scripts/5-index.py`

**Ce qu'il fait.** Il charge les 64 passages, calcule leur vecteur avec le modèle `BAAI/bge-small-en-v1.5` (à partir du titre du document, du titre de section et du texte) et les stocke dans une base Qdrant locale avec toutes leurs métadonnées. Chaque passage a un identifiant stable et une empreinte de son texte : à chaque exécution, seuls les passages nouveaux ou modifiés sont réindexés. Le script contient aussi la fonction de recherche, qui applique un filtre sur le statut (`en_vigueur`) et sur le rôle autorisé avant le classement, et il peut lancer une question de test avec `--test` et `--role`.

**Pourquoi.** L'indexation rend le corpus interrogeable par le sens et pas seulement par les mots. Le filtrage dans la requête est la mesure de sécurité centrale : un document interdit n'est jamais récupéré. L'indexation incrémentale répond à l'exigence de mises à jour du sujet, puisqu'une modification de document ne force pas à tout recalculer.

**Entrée / sortie.** Entrées : `data/processed/passages.jsonl` et `metadata.csv`. Sortie : la base vectorielle dans `data/qdrant/`.

### `scripts/6-eval_baseline.py`

**Ce qu'il fait.** Il prend les questions du split choisi (test par défaut), les transforme en vecteurs, les pose à la base avec le rôle `Administrator` et récupère les 10 meilleurs passages. Pour chaque question, il note le rang du bon passage (celui de la réponse d'origine) et du bon document, puis calcule Recall@1, 3, 5, 10 et MRR@10 aux deux niveaux. Il affiche enfin les questions dont le bon passage n'est pas en première position.

**Pourquoi.** C'est la mesure de départ, la baseline, à laquelle le reranker sera comparé. Sans elle, on ne pourrait pas dire si le fine-tuning améliore quoi que ce soit. La liste des erreurs montre quels thèmes sont confondus, ce qui guide le choix des exemples difficiles pour l'entraînement du reranker. Le rôle `Administrator` est utilisé pour mesurer la recherche seule, sans effet du filtrage par droits.

**Entrée / sortie.** Entrées : `data/processed/questions_split.csv` et la base Qdrant. Sorties : `data/eval/baseline_test.csv` (une ligne par question avec le rang du bon passage) et `data/eval/baseline_test.json` (les scores résumés).

### Fichiers de configuration

`requirements.txt` liste les dépendances Python pour que quelqu'un d'autre puisse recréer l'environnement avec une seule commande. `.gitignore` empêche de versionner l'environnement virtuel, les secrets, la base Qdrant (régénérable) et les modèles lourds. `.env` contient la clé `GEMINI_API_KEY` et ne doit jamais être publié, et `.env.example` montre quelle variable est attendue sans révéler sa valeur.

### Fichiers de données

`data/raw/hr_dataset_full.csv` est le dataset brut. Dans `data/processed/`, `answers_by_document.csv`, `document_plan.csv` et `questions_split.csv` décrivent le regroupement et le split, tandis que `metadata.csv` et `passages.jsonl` décrivent les documents et les passages du corpus. `data/corpus/` contient les 17 documents Markdown, `data/qdrant/` la base vectorielle et `data/eval/` les résultats d'évaluation.

## Données

La source est le dataset public [`adarsh-08/hr-policy-dataset`](https://huggingface.co/datasets/adarsh-08/hr-policy-dataset) (Hugging Face, licence Apache-2.0) : 512 paires question-réponse, soit 64 réponses de politique RH uniques, chacune reformulée par 8 questions. Aucun texte n'a été généré par le projet : chaque passage du corpus est une réponse du dataset recopiée sans modification. Ce que le projet ajoute, ce sont l'organisation en 17 documents (HR-01 à HR-12, FIN-01 à FIN-03, IT-01, POL-01), les titres de section, les départements, les niveaux de confidentialité, les rôles autorisés, la version V1 et une date de création fictive (2025-01-01).

Le découpage entraînement/test se fait par réponse et non par question, pour qu'une même réponse n'apparaisse jamais des deux côtés. Il donne 336 questions d'entraînement (42 réponses) et 176 questions de test (22 réponses), soit environ 34 % de test, graine 42.

## Rôles et contrôle d'accès

Cinq rôles existent : `Employee`, `HR Manager`, `IT Administrator`, `Finance Manager` et `Administrator`. Trois niveaux de confidentialité sont utilisés : `public_interne`, `restreint` et `confidentiel`. Quatorze documents sont ouverts à tous les rôles. HR-06 (recrutement) et HR-08 (révisions de salaire et plans d'amélioration de la performance) sont réservés à `HR Manager` et `Administrator`, et FIN-02 (primes et incitations) à `HR Manager`, `Finance Manager` et `Administrator`.

## Évaluation

Le retrieval est évalué sur les 176 questions de test avec le rôle `Administrator`, pour mesurer la recherche seule sans effet du filtrage. Le bon passage est celui de la réponse d'origine de la question. Le Recall@k est la part des questions dont le bon passage figure dans les k premiers résultats, et le MRR est la moyenne de l'inverse du rang du bon passage.

Résultats de la baseline (modèle `BAAI/bge-small-en-v1.5`, sans reranker). Au niveau passage : Recall@1 = 0,756, Recall@3 = 0,977, Recall@5 = 0,994, Recall@10 = 1,0 et MRR@10 = 0,864. Au niveau document : Recall@1 = 0,869, Recall@3 = 0,977, Recall@5 = 1,0 et MRR@10 = 0,927. Sur les 176 questions, 43 n'ont pas le bon passage en première position, avec des confusions entre thèmes voisins (par exemple le télétravail et les appareils de l'entreprise). Le bon passage étant toujours dans les 10 premiers résultats, un reranker peut en théorie atteindre un Recall@1 de 100 %.

## Serveur MCP (prévu)

Le serveur exposera trois outils : `search_documents` (recherche filtrée selon l'utilisateur), `get_user_permissions` (rôle et accès d'un utilisateur) et `get_document_metadata` (titre, version, statut et confidentialité d'un document, si l'utilisateur y a accès). Le rôle est toujours retrouvé côté serveur à partir de l'identifiant d'utilisateur, jamais pris dans un paramètre fourni par le modèle. Un script client servira à la démonstration.

## État d'avancement

Sont terminés : le téléchargement et la vérification du dataset, la préparation des données et le split, la génération du corpus V1, l'indexation dans Qdrant avec filtrage par rôle et mise à jour incrémentale, et l'évaluation de la baseline.

Restent à faire : la création des données synthétiques (utilisateurs fictifs, versions V2 de documents, tests d'accès, questions sans réponse dans le corpus), la comparaison d'un reranker pré-entraîné puis fine-tuné avec la baseline, la génération des réponses avec Gemini, le serveur MCP et son client de démonstration, l'évaluation finale sur tous les jeux de questions et le rapport.

## Limites connues

Les questions du dataset sont des reformulations de leurs réponses, ce qui rend le retrieval plus facile qu'en situation réelle et peut donner des scores flatteurs. Le corpus ne contient que 64 passages et des données synthétiques. Certaines questions ne sont pas couvertes par leur réponse, et plusieurs réponses se recoupent (par exemple le total des congés payés avec les trois types de congés, ou les réponses sur le remboursement des frais de voyage). Les données ajoutées par le projet pour tester les droits et les versions ne remplacent pas des cas réels.

## Crédits et licence

Dataset : `adarsh-08/hr-policy-dataset`, licence Apache-2.0. Modèle d'embedding : `BAAI/bge-small-en-v1.5`. Base vectorielle : Qdrant. Génération : Gemini. Licence du projet : à définir.
