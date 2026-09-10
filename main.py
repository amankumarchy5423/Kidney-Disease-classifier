from src.amanClassifier.logging.logger import logger
from src.amanClassifier.pipeline.project_pipeline import ProjectPipeline


try:
    obj_Project_pipeline = ProjectPipeline()
    obj_Project_pipeline.initiate_project_pipeline()
except Exception as e :
    logger.error("e")
    raise e