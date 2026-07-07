from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )
    }
    return models