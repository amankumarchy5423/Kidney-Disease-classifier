import os
import sys
from pathlib import Path

from src.amanClassifier.constants import *
from src.amanClassifier.logging.logger import logger
from src.amanClassifier.utils.common import *


params = load_yaml(file_path = Path("params.yaml"))

class DataIngestionConfig:

    def __init__(self):
        try:
            self.output_artifact_path = os.path.join(
                ARTIFACT_DIR,
                DATA_INGESTION_DIR
            )

            self.allowed_extensions = {
                ".jpg",
                ".jpeg",
                ".png"
            }

            self.img_size = params.data_ingestion.image_size

            self.batch_size = params.data_ingestion.batch_size

            self.validation_split = (
                params.data_ingestion.validation_split
            )

            self.validation_data_split = (
                params.data_ingestion.validation_data_split
            )

            self.seed = params.data_ingestion.seed

            self.Data_file_path = DATA_FILE_PATH

            self.output_train_data = os.path.join(
                self.output_artifact_path,
                TRAIN_FILE
            )

            self.output_test_data = os.path.join(
                self.output_artifact_path,
                TEST_FILE
            )

            self.output_val_data = os.path.join(
                self.output_artifact_path,
                VAL_FILE
            )

            create_directories([
                self.output_artifact_path,
                self.output_train_data,
                self.output_test_data,
                self.output_val_data
            ])

        except Exception as e:
            logger.info(e)
            raise e
        
class ModelBuildingConfig:
    def __init__(self):
        try:
            self.model_building_dir = os.path.join(ARTIFACT_DIR,MODEL_BUILDING)

            create_directories([self.model_building_dir])
        except Exception as e :
            logger.error(e)
            raise e

class ModelTrainingConfig:
    def __init__(self):
        try:
            self.model_thresold : float = 0.5
            self.model_training_dir : Path = os.path.join(ARTIFACT_DIR,MODEL_TRAINING)
            self.model_png : Path = os.path.join(self.model_training_dir , MODEL_STUCTURE_PNG)
            self.trained_model_file_path = os.path.join(self.model_training_dir , MODEL_FILE)

            create_directories([self.model_training_dir])
        except Exception as e:
            logger.error(e)
            raise e
        
# class ModelEvaluationConfig:
#     def __init__(self):
#         try:
            
#             self.bucket_name = common_variable.BUCKET_NAME
#             self.model_key = "models/ml_model/model.joblib"
#             self.pre_model_key = "models/ml_model/preprocessor.joblib"
#         except Exception as e :
#             my_log.error(e)
#             raise MyException(e,sys)

# # my_model\preprocessor.joblib