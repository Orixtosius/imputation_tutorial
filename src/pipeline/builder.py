from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.preprocessing._encoders import _BaseEncoder
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.utils.dict_of_list_ops import DictOfListOperator


class PipelineConstructor:

    def __init__(self, config: list[tuple[str, str, dict[str, any], tuple[str]]]) -> None:
        self.config = config
        self.preprocess_operator = DictOfListOperator()
        self.__group_by_transformers()
    
    def __group_by_transformers(self) -> None:
        for step in self.config:
            key = step[0]
            step_config = step[1:]
            if key.startswith("p"):
                self.preprocess_operator.__call__(key, step_config)
            elif key.startswith("e"):
                self.estimator_config = (key, step_config)

    def build(self) -> Pipeline:
        grouped_preprocess_config = self.preprocess_operator.get_collection()

        transformer_config = [self.__extract_transformer_for_feature(key, steps) 
                              for key, steps in grouped_preprocess_config.items()]

        preprocessor = ColumnTransformer(transformers=transformer_config, remainder="passthrough")
        estimator = self.__extract_estimator(self.estimator_config[1])

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            estimator
        ])

        return pipeline
    
    def __extract_steps_for_feature(self, step_config: tuple[str, dict]) -> tuple[str, _BaseEncoder]:
        preprocessor, params = step_config
        preprocessor_name = preprocessor.lower()
        extracted_step = (preprocessor_name, eval(f"{preprocessor}(**{params})"))
        return extracted_step
    
    def __extract_transformer_for_feature(self, key: str, steps: list[tuple[str, dict, tuple]]) -> tuple[str, _BaseEncoder, tuple]:
        features = steps[0][-1]
        pipeline_steps = [self.__extract_steps_for_feature(s[:-1]) for s in steps]
        transformer = Pipeline(steps=pipeline_steps)
        transformer_step = (key, transformer, features)
        return transformer_step
    
    def __extract_estimator(self, step: tuple[str, dict]) -> tuple[str, BaseEstimator]:
        estimator = self.__extract_steps_for_feature(step)
        return estimator
