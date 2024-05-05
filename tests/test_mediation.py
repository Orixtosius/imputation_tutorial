import unittest
from sklearn.datasets import fetch_openml
from context import src
from src.mediator import TrainingMediator
from config import test_config


class TestMediator(unittest.TestCase):
    def setUp(self) -> None:
        self.mediator = TrainingMediator(test_config)

    def test_mediation(self):
        X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
        X = X[["pclass", "sex", "sibsp", "parch", "age", "fare"]]
        X = X.astype({
            "pclass": str, "sex": str, "sibsp": str, "parch": str,
            "age": float, "fare": float
        })
        y = y.astype(float)
        scores = self.mediator(X, y)
        print(scores)