from src.amanClassifier.logging.logger import logger
from src.amanClassifier.constants import PARAMS_FILE_PATH
from src.amanClassifier.utils.common import (
    load_yaml,
    save_binary_file
)
from src.amanClassifier.config.configuration import ModelEvaluationConfig
from src.amanClassifier.Artifact.project_artifact import ModelTrainingArtifact
import mlflow
import mlflow.keras


class ModelEvaluation:
    def __init__(self,config : ModelEvaluationConfig , artifact : ModelTrainingArtifact):
        self.config = config
        self.artifact = artifact
        self.params = load_yaml('params.yaml').model_evaluation

    def check_model_accuracy(self,history):
        try:
            key = False
            if (self.params.at_least_training_accuracy <= history["accuracy"][-1] and 
            self.params.at_least_val_accuracy <= history['val_accuracy'][-1]):
                key = True

            logger.info(f"eveluation output of accuracy is {key}")

            return key
        except Exception as e :
            logger.exception(e)
            raise e

    def check_model_loss(self):
        try:
            pass
        except Exception as e :
            logger.exception(e)
            raise e 

    def send_model_into_production(self):
        try:
            pass
        except Exception as e :
            logger.exception(e)
            raise e
    