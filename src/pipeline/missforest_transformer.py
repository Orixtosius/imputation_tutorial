from sklearn.base import TransformerMixin, BaseEstimator
from missingpy import MissForest
from scipy import sparse
import numpy as np
from sklearn.impute import SimpleImputer


class MissForestImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = MissForest()

    def fit(self, X, y=None):
        X_formatted = X.astype(float)
        if sparse.issparse(X):
            X_formatted = X_formatted.toarray()
        self.imputer.fit(X_formatted, y)
        return self

    def transform(self, X):
        X_formatted = X.astype(float)        
        if sparse.issparse(X_formatted):
            X_formatted = X_formatted.toarray()
        X_imputed = self.imputer.transform(X_formatted)
        return X_imputed