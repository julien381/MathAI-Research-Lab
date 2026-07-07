import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from model import get_models
import joblib

# On importe les fonctions qu'on vient de ranger dans data.py
from data import load_data, split_features_target
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# ============================
# 1. Chargement des données
# ============================
print("--- CHARGEMENT DES DONNÉES ---")
df = load_data("../data/raw/creditcard.csv")
X, y = split_features_target(df)

print(f"Dataset chargé avec succès ! Dimensions globales : {df.shape}")



print("\n--- SÉPARATION TRAIN / TEST ---")
# test_size fait la découpe : 20% pour l'examen final (et 80% pour l'entraînement).
# 'stratify=y' garde exactement la même proportion de fraudes dans les deux blocs.
# 'random_state=42' fige le mélange pour obtenir les mêmes résultats à chaque lancement.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y 
)

print(f"Taille du bloc d'Entraînement (Train) : {X_train.shape}")
print(f"Taille du bloc d'Évaluation (Test)    : {X_test.shape}")


# ============================
# 3. Normalisation (Régler le cas du super-héros)
# ============================
print("\n--- NORMALISATION (StandardScaler) ---")

# Crée l'outil pour remettre toutes les variables à la même échelle (éviter que les gros chiffres masquent les petits).
scaler = StandardScaler()

# fit_transform : calcule la moyenne sur le Train et rétrécit 'Time' et 'Amount' pour les mettre au même niveau que les autres.
X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])

# transform : applique le MÊME rétrécissement sur le Test, sans recalculer la moyenne pour ne pas tricher sur l'examen.
X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

print("Normalisation de 'Time' et 'Amount' effectuée.")


# ============================
# 4. SMOTE (Équilibrage des billes rouges)
# ============================
print("\n--- ÉQUILIBRAGE AVEC SMOTE ---")
print("Nombre de fraudes AVANT SMOTE dans le jeu d'entraînement :")
print(y_train.value_counts())# Affiche le déséquilibre de départ (très peu de fraudes).

# Prépare le cloneur en figeant le hasard pour avoir toujours les mêmes résultats.
smote = SMOTE(random_state=42)

# SMOTE crée des clones de fraudes UNIQUEMENT dans le bloc d'entraînement pour obtenir une égalité 50/50.
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nNombre de fraudes APRÈS SMOTE dans le jeu d'entraînement :")
print(y_train_resampled.value_counts())# Vérifie que le nombre de fraudes est maintenant égal aux normales


# ============================
# 5. Dimensions Finales
# ============================
print("\n--- DIMENSIONS FINALES DES DONNÉES ---")
# Affiche la taille finale du bloc d'entraînement (qui a gonflé avec les clones).
print(f"X_train (prêt pour l'IA) : {X_train_resampled.shape}")
# Affiche la taille du bloc de test (qui lui n'a pas bougé et reste propre, sans clones).
print(f"X_test (prêt pour le test) : {X_test.shape}")

# ============================
# 6. Entraînement et Sélection du Meilleur Modèle
# ============================
print("\n==============================")
print("Entraînement des modèles")
print("==============================")

# Appel de la fonction catalogue pour lister tous les modèles à tester
models = get_models()

# Initialisation des variables pour suivre le modèle le plus performant
best_model = None
best_score = 0

# Boucle : on itère sur chaque modèle présent dans notre catalogue (Logistic, RF, XGB)
for name, model in models.items():

    print(f"\n{name}")

    # Phase d'apprentissage : l'IA apprend à reconnaître les fraudes 
    # sur les données équilibrées (SMOTE)
    model.fit(
        X_train_resampled,
        y_train_resampled
    )

    # Phase de test : l'IA fait des prédictions sur les données brutes (jamais vues)
    predictions = model.predict(X_test)

    # Calcul des métriques de performance
    precision = precision_score(y_test, predictions)# Fiabilité : évite ou non les fausses alertes
    recall = recall_score(y_test, predictions)# Sensibilité : proportion de vrais voleurs attrapés
    f1 = f1_score(y_test, predictions)# Équilibre : le juste milieu entre fiabilité et sensibilité
    roc = roc_auc_score(y_test, predictions)# Note globale : capacité à distinguer l'honnête du fraudeur

    # Affichage des scores dans la console pour comparer les résultats en direct
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc:.4f}")

    # Logique de sélection : si le modèle actuel a un meilleur ROC-AUC 
    # que le précédent, il devient notre nouveau "meilleur modèle"
    if roc > best_score:
        best_score = roc
        best_model = model

# ============================
# 7. Sauvegarde sur le disque
# ============================
# Enregistre le gagnant dans le dossier models/
joblib.dump(
    best_model,
    "../models/best_model.pkl"
)

print("\nMeilleur modèle sauvegardé.")


# ==============================================================================
# ÉTAPE 6 : ÉVALUATION ET GRAPHIQUES
# ==============================================================================
print("\n--- GÉNÉRATION DES GRAPHES D'ÉVALUATION ---")
import os
os.makedirs("../reports", exist_ok=True)

# 6.1 Matrice de Confusion : Visualiser les vrais positifs et les fausses alertes
plt.figure(figsize=(6, 6))
ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test,
    y_test,
    cmap="Blues"
)
plt.title(f"Matrice de Confusion - {type(best_model).__name__}")
plt.savefig("../reports/confusion_matrix.png")
plt.close() # Ferme la figure pour libérer la mémoire
print("✔ Matrice de confusion sauvegardée dans /reports")

# 6.2 Courbe ROC : courbe de performance globale du modèle
plt.figure(figsize=(6, 6))
RocCurveDisplay.from_estimator(
    best_model,
    X_test,
    y_test
)
plt.title(f"Courbe ROC - {type(best_model).__name__}")
plt.savefig("../reports/roc_curve.png")
plt.close()
print("✔ Courbe ROC sauvegardée dans /reports")

# 6.3 Importance des variables : poids de chaque caractéristique dans la décision
print("--- ANALYSE DE L'IMPORTANCE DES VARIABLES ---")

# Si le gagnant est un arbre (Random Forest / XGBoost)
if hasattr(best_model, "feature_importances_"):
    importance_values = best_model.feature_importances_
# Si le gagnant est linéaire (Régression Logistique), on extrait la valeur absolue des coefficients
elif hasattr(best_model, "coef_"):
    importance_values = abs(best_model.coef_[0])
else:
    importance_values = None

if importance_values is not None:
    # Structuration des poids dans un DataFrame trié par ordre décroissant
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance_values
    }).sort_values(by="Importance", ascending=False)

    # Tracé du diagramme à barres horizontales pour le Top 15
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df["Feature"][:15], importance_df["Importance"][:15], color="skyblue")
    plt.gca().invert_yaxis()  # Met la variable la plus critique tout en haut
    plt.title(f"Top 15 des variables déterminantes - {type(best_model).__name__}")
    plt.tight_layout()
    plt.savefig("../reports/feature_importance.png")
    plt.close()
    print("✔ Graphique des importances sauvegardé dans /reports")


# 6.4 SHAP : Analyse locale et directionnelle de l'impact des variables
print("--- CALCUL DES VALEURS SHAP ---")
import shap

# Initialisation de l'explicateur SHAP adapté selon la nature du modèle gagnant
if "Linear" in type(best_model).__name__ or "Logistic" in type(best_model).__name__:
    explainer = shap.LinearExplainer(best_model, X_train_resampled)
    shap_values = explainer.shap_values(X_test)
else:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

# Tracé du summary plot (nuage de points d'impact global)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title(f"Résumé SHAP - {type(best_model).__name__}", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("../reports/shap_summary.png")
plt.close()
print("✔ Graphique SHAP sauvegardé dans /reports")
print("\nToutes les analyses de l'Étape 6 ont été générées avec succès !")
