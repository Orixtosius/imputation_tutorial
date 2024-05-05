from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator


class Evaluator:

    def __call__(self, estimator: BaseEstimator, X, y):
        metrics = ('r2', 'neg_mean_squared_error')
        cv_no = 5
        scores = cross_validate(estimator, X, y, cv=cv_no, scoring=metrics, return_train_score=True, error_score='raise')
        return scores