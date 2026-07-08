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

---

# Guide Technique - MathAI Fraud Detection API

## 1. Lancement de l'application
Exécuter la commande suivante dans le terminal pour construire l'image et lancer le conteneur Docker :
docker-compose up --build

## 2. Accès à l'interface de test (Lien à mettre dans le navigateur)
Ouvrir votre navigateur internet (Chrome, Firefox, Edge, etc.) et copier-coller l'adresse URL suivante dans la barre de recherche :
http://localhost:8000/docs

## 3. Utilisation de l'interface et soumission du test
Dérouler la route POST /predict en cliquant dessus.
Cliquer sur le bouton Try it out.
Remplacer le contenu de la case Request body par le vecteur (le tableau de données) de 30 variables ci-dessous.
Cliquer sur le bouton bleu Execute pour obtenir le résultat de la prédiction.

## 4. Exemple de vecteur (30 variables) à saisir
{
  "features": [
    0.0, -0.425966, 0.960523, 1.141822, -0.168252, 0.420987, -0.029728, 
    0.473012, 0.139421, -0.210533, -0.452110, 0.682341, -0.125430, 
    -0.852104, 0.215390, 0.914210, -0.312054, -0.115432, 0.085421, 
    -0.089431, 0.051243, -0.241053, -0.612054, -0.052140, -0.352104, 
    -0.112043, 0.412053, 0.152104, 0.042105, 45.99
  ]
}

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
