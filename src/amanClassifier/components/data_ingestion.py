from src.amanClassifier.logging.logger import logger
from src.amanClassifier.constants import *
from src.amanClassifier.utils.common import create_directories
from src.amanClassifier.Artifact.project_artifact import DataIngestionArtifact
from src.amanClassifier.config.configuration import DataIngestionConfig

import os
from pathlib import Path
import numpy as np
from PIL import Image


class DataIngestion:
    def __init__(self,config:DataIngestionConfig)-> None:
        try:
            self.config = config
            
        except Exception as e :
            logger.info(e)
            raise e

    def load_images_and_check_extensions(self):
        try:
            pass
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e

    def resize_images(self, img_path: Path, img_size: tuple): 
        try:
            pass
        except Exception as e:
            logger.error(f"Error resizing image {img_path}: {e}")
            raise e

    def convert_images_to_array_and_grayscale(self, img_path: Path):
        try:
            pass
        except Exception as e:
            logger.error(f"Error converting image to array and grayscale {img_path}: {e}")
            raise e
    def save_images_in_data_ingestion_artifact(self, img_array: np.ndarray, output_path: Path):
        try:
            pass
        except Exception as e:
            logger.error(f"Error saving image array to {output_path}: {e}")
            raise e

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            logger.info("<____ Starting data ingestion process ____>")
            obj_data_ingestion_config = DataIngestionConfig()
            obj_data_ingestion = DataIngestion(config = obj_data_ingestion_config)
            logger.info(f"Data ingestion config: {obj_data_ingestion_config.__dict__}")

            loaded_images = obj_data_ingestion.load_images_and_check_extensions()

            resized_images = obj_data_ingestion.resize_images(img_path = , img_size=obj_data_ingestion_config.img_size) 
            image_arrays = [obj_data_ingestion.convert_images_to_array_and_grayscale(img_path) for img_path in resized_images]
            output_dir = obj_data_ingestion_config.Data_file_path
            create_directories([output_dir])

            logger.info("<____ End data ingestion process ____>")
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise e