import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

SEED = 42

def get_stratified_folds(df, target_col="Label", n_splits=5):
    """
    Sınıf oranlarını koruyarak 5-fold cross validation indeksleri üretir.
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED
    )
    return list(skf.split(df, df[target_col]))


def evaluate_predictions(y_true, y_pred_bin):
    """
    Yarışma metriği olan F1 skorunu hesaplar.
    """
    return f1_score(y_true, y_pred_bin, zero_division=0)
