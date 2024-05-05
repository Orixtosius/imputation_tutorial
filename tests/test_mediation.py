import unittest
from sklearn.datasets import fetch_openml
from context import src
from src.mediator import TrainingMediator
from config import test_config, test_config_3


class TestMediator(unittest.TestCase):
    def setUp(self) -> None:
        self.mediator = TrainingMediator(test_config)
        self.mediator2 = TrainingMediator(test_config_3)

    def __format_input_data(self):
        X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
        X = X[["pclass", "sex", "sibsp", "parch", "age", "fare"]]
        X = X.astype({
            "pclass": str, "sex": str, "sibsp": str, "parch": str,
            "age": float, "fare": float
        })
        y = y.astype(float)
        return X, y

    def test_mediation(self):
        X, y = self.__format_input_data()
        scores = self.mediator(X, y)
        print(scores)

    def test_mediation_case_2(self):
        X, y = self.__format_input_data()
        scores = self.mediator2(X, y)
        print(scores)