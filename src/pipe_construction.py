from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder


class PipelineConstructor:

    def __init__(self, config: list[tuple]) -> None:
        self.config = config
    
    def build(self):
        steps = [eval(f"{step[0]}(**{step[1]})") for step in self.config]
        return Pipeline(steps)