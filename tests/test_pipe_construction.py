import unittest
from src.pipe_construction import PipelineConstructor

class TestConstructor(unittest.TestCase):

    def setUp(self) -> None:
        pipeline_config = [
            ("RobustScaler", {"with_centering": False}),
            ("OneHotEncoder", {"handle_unknown": 'ignore'})
        ]
        self.constructor = PipelineConstructor(pipeline_config)

    def test_building(self):
        pipeline = self.constructor.build()