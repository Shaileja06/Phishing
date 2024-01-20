import joblib
from tpot import TPOTClassifier

def loadmodel(dir):
    """Loads a saved model from the specified directory."""
    return joblib.load(dir)

def train_model(X, y):
    tpot_config = {
        'xgboost.XGBClassifier': {
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'n_estimators': range(50, 500),
            'max_depth': range(3, 10),
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'min_child_weight': range(1, 10)
        }
    }

    tpot = TPOTClassifier(generations=5, population_size=20, config_dict=tpot_config, verbosity=2, random_state=42, scoring='accuracy')
    tpot.fit(X, y)
    return tpot
