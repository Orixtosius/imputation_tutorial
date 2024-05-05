import unittest
from sklearn.preprocessing import RobustScaler
from src.pipe_construction import PipelineConstructor



pipeline_config = [
            ("RobustScaler", {"with_centering": False}),
            ("OneHotEncoder", {"handle_unknown": 'ignore'})
        ]
constructor = PipelineConstructor(pipeline_config)
constructor.build()