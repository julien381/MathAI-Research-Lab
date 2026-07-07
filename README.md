# MathAI-Research-Lab

## Projet : Détection de fraude bancaire par IA

Ce projet a pour objectif de construire un système de détection de fraude bancaire à partir de données de transactions.

---

## Problématique

Les fraudes bancaires représentent un enjeu majeur pour les institutions financières.

Le but est de construire un modèle de Machine Learning capable de :
- Détecter les transactions frauduleuses
- Gérer un dataset très déséquilibré
- Minimiser les faux négatifs (fraudes non détectées)

---

## Type de problème

- Classification binaire :
  - 0 = Transaction normale
  - 1 = Fraude
- Problème fortement déséquilibré (0,17% de fraudes)

---

## Approche Technique

- Analyse exploratoire des données (EDA)
- Prétraitement des données (Stratification et StandardScaler)
- Gestion du déséquilibre via l'algorithme SMOTE
- Modèles de Machine Learning : Régression logistique, Random Forest, XGBoost
- Évaluation et Interprétabilité : F1-score, ROC-AUC, Matrice de confusion et SHAP
- Déploiement : API Web avec FastAPI et Pydantic
- Conteneurisation : Image isolée via Docker et Docker Compose

---

## Données

1. Téléchargez le dataset à l'adresse suivante : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Placez le fichier "creditcard.csv" dans le répertoire "data/raw/".

## Structure du projet

```text
MathAI-Research-Lab/
├── app/                  # Code du micro-service API
│   ├── api.py            # Logique de prédiction FastAPI
│   └── schemas.py        # Validation des données Pydantic
├── data/
│   └── raw/              # Fichier creditcard.csv
├── models/               # Stockage du modèle persistant (.pkl)
├── src/                  # Scripts de traitement
│   ├── data.py           # Pipeline de chargement des données
│   └── train.py          # Script d'apprentissage
├── Dockerfile            # Configuration de l'image Docker
├── docker-compose.yml    # Orchestration du conteneur
└── requirements.txt      # Dépendances logicielles
