import unittest
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from context import src
from src.pipeline.builder import PipelineConstructor
from config import test_config, test_config_2


class TestConstructor(unittest.TestCase):
    pass
    def setUp(self) -> None:
        self.constructor = PipelineConstructor(test_config_2)
        self.pipeline = self.constructor.build()
        self.steps = self.pipeline.steps

    def __setUp_for_case_1(self):
        self.constructor = PipelineConstructor(test_config_2)
        self.pipeline = self.constructor.build()
        return self.pipeline.steps
    
    def __setUp_for_case_2(self):
        self.constructor = PipelineConstructor(test_config)
        self.pipeline = self.constructor.build()
        return self.pipeline.steps

    def test_building_first_step_name(self):
        steps = self.__setUp_for_case_1()
        self.assertEqual(steps[0][0], "preprocessor")

    def test_building_first_step_type(self):
        steps = self.__setUp_for_case_1()
        self.assertIsInstance(steps[0][1], ColumnTransformer)

    def test_building_transformer_steps(self):
        steps = self.__setUp_for_case_1()
        transformation_step: ColumnTransformer = steps[0][1]
        scaler_pipeline = transformation_step.get_params()["p1"]
        encoder_pipeline = transformation_step.get_params()["p2"]
        self.assertIsInstance(scaler_pipeline, Pipeline)
        self.assertIsInstance(encoder_pipeline, Pipeline)
        self.assertIsInstance(scaler_pipeline.steps[0][1], RobustScaler)
        self.assertIsInstance(encoder_pipeline.steps[0][1], OneHotEncoder)
        
    def test_estimator_step(self):
        steps = self.__setUp_for_case_1()
        estimator_step = steps[1][1]
        self.assertIsInstance(estimator_step, LinearRegression)

    def test_building_second_step_name(self):
        steps = self.__setUp_for_case_2()
        self.assertEqual(steps[0][0], "preprocessor")

    def test_building_second_step_type(self):
        steps = self.__setUp_for_case_2()
        self.assertIsInstance(steps[0][1], ColumnTransformer)