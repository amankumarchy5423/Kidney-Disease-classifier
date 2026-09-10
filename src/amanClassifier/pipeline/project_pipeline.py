from src.amanClassifier.logging.logger import logger

from src.amanClassifier.components.data_ingestion import DataIngestion
from src.amanClassifier.Artifact.project_artifact import DataIngestionArtifact
from src.amanClassifier.config.configuration import DataIngestionConfig

from src.amanClassifier.components.model_building import ModelBuilding
from src.amanClassifier.Artifact.project_artifact import ModelBuildingArtifact
from src.amanClassifier.config.configuration import ModelBuildingConfig

from src.amanClassifier.components.model_training import ModelTraining
from src.amanClassifier.config.configuration import ModelTrainingConfig






class ProjectPipeline:
    def __init__(self):
        pass
    
    def data_prepration_pipeline(self)->DataIngestionArtifact:
        try:
            logger.info("data ingestion pipeline starts.....")
            obj_data_ingestion_config = DataIngestionConfig()
            obj_data_ingestion = DataIngestion(config = obj_data_ingestion_config)
            out_data_ingestion = obj_data_ingestion.initiate_data_ingestion()
            logger.info("data ingestion pipeline ends.....")

            return out_data_ingestion
        except Exception as e :
            logger.error(e)
            raise(e)
    
    def model_building_pipeline(self,out_data_ingestion:DataIngestionArtifact)->ModelBuildingArtifact:
        try:
            logger.info("model_building_pipeline starts .....")

            obj_model_building_config = ModelBuildingConfig()
            obj_model_building = ModelBuilding(config = obj_model_building_config , artifact = out_data_ingestion)
            out_model_building = obj_model_building.initiate_model_building()

            return out_model_building

            logger.info("model_building_pipeline ends .....")
        except Exception as e:
            logger.error(e)
            raise(e)

    def model_training_pipeline(self,out_model_building : ModelBuildingArtifactArtifact,out_data_ingestion:DataIngestionArtifact):
        try:
            logger.info("model training pipeline started ......")

            obj_model_training_config = ModelTrainingConfig()
            obj_model_training = ModelTraining(config=obj_model_training_config, model_artifact=out_model_building, data_artifact=out_data_ingestion)
            obj_model_training.initiate_model_training()

            logger.info("model training pipeline ended ......")
        except Exception as e :
            logger.error(e)
            raise e

    def initiate_project_pipeline(self):
        try:
            data_ingestion_output = self.data_prepration_pipeline()
            logger.info("data ingestion pipeline output got ....")

            model_building_output = self.model_building_pipeline(out_data_ingestion=data_ingestion_output)
            logger.info("model building pipeline output got ")

            self.model_training_pipeline()
        except Exception as e :
            logger.error(e)
            raise e



if __name__ == '__main__':
    try:
        obj_Project_pipeline = ProjectPipeline()
        obj_Project_pipeline.initiate_project_pipeline()
    except Exception as e :
        logger.error("e")
        raise e