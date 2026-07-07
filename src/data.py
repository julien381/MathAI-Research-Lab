import pandas as pd

def load_data(path: str):
    """Charge le dataset CSV"""
    df = pd.read_csv(path)
    return df

def split_features_target(df: pd.DataFrame):
    """
    Sépare les variables explicatives (features) de la variable cible (target).

    Args:
        df (pd.DataFrame): Le dataset complet contenant les transactions.

    Returns:
        tuple: 
            - X (pd.DataFrame): Matrice des caractéristiques (sans la colonne 'Class').
            - y (pd.Series): Vecteur cible contenant les étiquettes de fraude (0 ou 1).
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y
