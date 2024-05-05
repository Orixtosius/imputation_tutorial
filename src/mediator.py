from src.pipeline.builder import PipelineConstructor
from src.evaluation import Evaluator


class TrainingMediator:
    
    def __init__(self, pipeline_configs: list) -> None:
        self.constructor = PipelineConstructor(pipeline_configs)
        self.evaluator = Evaluator()

    def __call__(self, X, y) -> dict:

        pipeline = self.constructor.build()
        scores = self.evaluator(pipeline, X, y)
        return scores