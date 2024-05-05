import unittest
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from context import src
from src.pipeline.builder import PipelineConstructor

class TestConstructor(unittest.TestCase):
    pass
    def setUp(self) -> None:
        pipeline_config = [
            ("p1", "RobustScaler", {"with_centering": False}, ("Column1", "Column2")),
            ("p2", "OneHotEncoder", {"handle_unknown": "ignore"}, ("Column3", "Column4")),
            ("e1", "LinearRegression", {})
        ]
        self.constructor = PipelineConstructor(pipeline_config)
        self.pipeline = self.constructor.build()
        self.first_step = self.pipeline.steps[0]
        self.second_step = self.pipeline.steps[1][1]

    def test_building_first_step(self):
        self.assertEqual(self.first_step[0], "preprocessor")
        self.assertIsInstance(self.first_step[1], ColumnTransformer)

    def test_first_step_params(self):
        param = self.first_step[1].get_params()["transformers"][0][1].steps[0][1].get_params()["with_centering"]
        self.assertFalse(param)

    def test_building_second_step(self):
        self.assertIsInstance(self.second_step, OneHotEncoder)

    def test_second_step_params(self):
        param = self.second_step.get_params()["handle_unknown"]
        self.assertEqual("ignore", param)