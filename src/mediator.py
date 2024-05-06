from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from src.pipeline.builder import PipelineConstructor
from src.evaluation import Evaluator


class TrainingMediator:
    
    def __init__(self) -> None:     
        self.score_cache = list()
        self.cv_no = 5
        self.metrics = ('r2', 'neg_mean_squared_error')

    def __call__(self, pipeline_configs: list, X, y, model_no: int) -> dict:
        self.constructor = PipelineConstructor(pipeline_configs)
        pipeline = self.constructor.build()
        scores = cross_validate(
            pipeline, 
            X, 
            y, 
            cv=self.cv_no, 
            scoring=self.metrics, 
            return_train_score=True, 
            error_score='raise'
        )
        self.score_cache.append((model_no, scores))
        return scores
    
    def choose_best_model(self, scores: dict) -> tuple[int, dict[str, float]]:
        temp_no = 0
        temp_scores = self.score_cache[0]
        
        for no, scores in self.score_cache[1:]:
            model_score = self.__calculate_model_performance(temp_scores, scores)
            if model_score < 0:
                temp_no = no
                temp_scores = model_score

        return temp_no, temp_scores

    def __calculate_model_performance(self, old_score: dict, new_score: dict) -> float:
        temp_overfit_dif = old_score['train_r2'] - old_score['test_r2']
        overfit_dif = new_score['train_r2'] - new_score['test_r2']
        
        if temp_overfit_dif < overfit_dif:
            return 1
        
        train_mse_dif = new_score['train_neg_mean_squared_error'] - old_score['train_neg_mean_squared_error']
        test_mse_dif = new_score['test_neg_mean_squared_error'] - old_score['test_neg_mean_squared_error']
        model_score = (2*test_mse_dif + train_mse_dif) / overfit_dif
        return model_score