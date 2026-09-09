from src.amanClassifier.logging,logger import logger
from src.amanClassifier.components.data_ingestion import DataIngestion
from src.amanClassifier.Artifact.project_artifact import DataIngestionArtifact
from src.amanClassifier.config.configuration import DataIngestionConfig




class ProjectPipeline:
    def__init__(self):
        pass
    
    def data_prepration_pipeline(self)->DataIngestionArtifact:
        logger.info("data ingestion pipeline starts.....")
        obj_data_ingestion_config = DataIngestionConfig()
        obj_data_ingestion = DataIngestion(config = obj_data_ingestion_config)
        out_data_ingestion = obj_data_ingestion.initiate_data_ingestion()
        logger.info("data ingestion pipeline ends.....")

        return out_data_ingestion



