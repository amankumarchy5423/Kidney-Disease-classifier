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
from tensorflow import keras
from mlflow import MlflowClient
from pathlib import Path


class ModelEvaluation:
    def __init__(self,config : ModelEvaluationConfig , artifact : ModelTrainingArtifact):
        self.config = config
        self.artifact = artifact
        self.params = load_yaml('params.yaml').model_evaluation
        self.client = MlflowClient()

    def check_model_accuracy(self,history):
        try:
            accuracy_report = False
            if (self.params.at_least_training_accuracy <= history["accuracy"][-1] and 
            self.params.at_least_val_accuracy <= history['val_accuracy'][-1]):
                accuracy_report = True

            logger.info(f"eveluation output of accuracy is {accuracy_report}")

            return accuracy_report
        except Exception as e :
            logger.exception(e)
            raise e

    def check_model_loss(self,history):
        try:
            loss_report = False
            if (self.params.at_most_val_loss >= history["val_loss"][-1] and 
                self.params.at_most_training_loss >= history["loss"][-1]):
                loss_report = True

            logger.info(f"eveluation output of loss is {loss_report}")

            return loss_report
        
        except Exception as e :
            logger.exception(e)
            raise e 

    def send_model_into_production(self,accuracy_report,loss_report):

        try:
    
            if not accuracy_report or not loss_report:
    
                logger.warning(
                    "Model failed evaluation. "
                    "Not promoting to production."
                )
    
                return False
    
            logger.info(
                "Model passed evaluation. "
                "Promoting model to production."
            )
    
            # Load candidate model
            model = self.artifact.best_model
            
    
           
            Path("model").mkdir(
                parents=True,
                exist_ok=True
            )
    
            model.save(self.config.production_path)
    
            logger.info(
                f"Production model saved: {self.config.production_path}"
            )
    
            model_info = mlflow.keras.log_model(
                model=model,
                name="kidney_classifier"
            )
    
            logger.info(
                f"MLflow model URI: {model_info.model_uri}"
            )
    
            registered_model = mlflow.register_model(
                model_uri=model_info.model_uri,
                name="KidneyClassifier"
            )
    
            logger.info(
                f"Registered model version: "
                f"{registered_model.version}"
            )
    
            self.client.set_registered_model_alias(
                name="KidneyClassifier",
                alias="Production",
                version=registered_model.version
            )
    
            logger.info(
                f"Model version {registered_model.version} "
                f"promoted to Production."
            )
    
            return registered_model
    
        except Exception as e:
    
            logger.exception(e)
            raise e

    def initiate_model_evaluation(self):
        try:
            history = self.artifact.history.history

            logger.info("model accuracy checking starts ...")
            accuracy_report  = self.check_model_accuracy(history=history)
            logger.info("model accuracy checking ends ...")

            logger.info("model loss calculation starts ...")
            loss_report = self.check_model_loss(history=history)
            logger.info("model loss calculation ends ...")

            logger.info("sending model to production starts ...")
            info_registered_model = self.send_model_into_production(accuracy_report=accuracy_report,
                                            loss_report=loss_report)
            logger.info("sending model to production ends ...")
            

        except Exception as e:
            logger.exception(e)
            raise e