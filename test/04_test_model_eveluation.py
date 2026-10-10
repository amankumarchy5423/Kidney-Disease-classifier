from src.amanClassifier.logging.logger import logger
from src.amanClassifier.components.model_evaluation import ModelEvaluation
import pytest



model_evaluation = ModelEvaluation.__new__(ModelEvaluation)

def test_model_accuracy():
    model_evaluation.check_model_accuracy