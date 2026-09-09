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

    def load_images_and_check_extensions(self,image_path : Path):
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
    def save_images_in_data_ingestion_artifact(self, img_array: np.ndarray, data_ingestion_artifact: Path):
        try:
            pass
        except Exception as e:
            logger.error(f"Error saving image array to {output_path}: {e}")
            raise e

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            logger.info("<____ Starting data ingestion process ____>")
            
            logger.info("image loading and their extension checking is started....")
            output_path1 = self.load_images_and_check_extensions(image_path = self.config.Data_file_path)
            logger.info("image loading and their extension checking is ended....")

            logger.info("image resizing starts....")
            output_path2 = self.resize_images(image_path = output_path1,img_size = self.config.img_size)
            logger.info("image resizing ends....")

            logger.info("converting image to array and grayscale is starts.....")
            output_path3 = self.convert_images_to_array_and_grayscale(img_path = output_path2)
            logger.info("converting image to array and grayscale is ends.....")

            logger.info("image is saving ....")
            self.save_images_in_data_ingestion_artifact(img_array = output_path3,data_ingestion_artifact = self.config.output_artifact_path)
            logger.info(f"image is saved at {self.config.output_artifact_path} ....")

            return DataIngestionArtifact(data_ingestion_artifact = self.config.output_artifact_path)



            logger.info("<____ End data ingestion process ____>")
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise e