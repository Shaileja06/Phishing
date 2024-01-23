import joblib
from tpot import TPOTClassifier
import pandas as pd

def loadmodel(dir):
    """Loads a saved model from the specified directory."""
    return joblib.load(dir)

