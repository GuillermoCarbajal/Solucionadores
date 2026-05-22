from sklearn.model_selection import KFold
import numpy as np
from sklearn.base import clone

def oof_prediction(X, y, clf):
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scores_oof = np.empty(len(y))
    scores_oof[:] = np.nan

    for train_idx, val_idx in kf.split(X):
        model = clone(clf)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        scores_oof[val_idx] = model.predict_proba(X.iloc[val_idx])
    
    return scores_oof