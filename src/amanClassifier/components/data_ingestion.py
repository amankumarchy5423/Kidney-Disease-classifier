from src.amanClassifier.logging.logger import logger
from src.amanClassifier.constants import *
from src.amanClassifier.utils.common import create_directories
from src.amanClassifier.Artifact.project_artifact import DataIngestionArtifact
from src.amanClassifier.config.configuration import DataIngestionConfig

import os
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory

class DataIngestion:
    def __init__(self,config:DataIngestionConfig)-> None:
        try:
            self.config = config
            
        except Exception as e :
            logger.info(e)
            raise e

    def load_train_and_test_data(self,image_path : Path):
        try:
            train_data = image_dataset_from_directory(
                directory = config.Data_file_path,
                labels = 'inferred',
                batch_size = 32,
                label_mode='categorical',
                image_size = (256,256),
                validation_split = 0.2,
                subset = 'training',
                seed = 42
            )
            logger.info("training data loaded.....")

            test_data = image_dataset_from_directory(
                directory = config.Data_file_path,
                labels = 'inferred',
                batch_size = 32,
                label_mode='categorical',
                image_size = (256,256),
                validation_split = 0.2,
                subset = 'validation',
                seed = 42
            )
            logger.info("testing data loaded.....")

            return train_data , test_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e

    def normalization_process(self,image,label):
        try:
            image = tf.cast(image / 255 , tf.float32)
            return image , label
        except Exception as e :
            logger.error(e)
            raise e

    def normalize_images(self, train_img: any, test_img : any): 
        try:
            train_img = train_img.map(self.normalization_process)
            test_img = test_img.map(self.normalization_process)

            return train_img , test_img
        except Exception as e:
            logger.error(f"Error resizing image {img_path}: {e}")
            raise e

    def skip_corrupted_image_and_performance_optimization(self, train_img:any , test_img:any):
        try:
            train_img = train_img.ignore_errors()
            test_img = test_img.ignore_errors()
            logger.info("corrupted image is deleted automatically......")

            AUTOTUNE = tf.data.AUTOTUNE
            train_img = train_img.prefetch(AUTOTUNE)
            test_img = test_img.prefetch(AUTOTUNE)
            logger.info("performance optimization completed....")

            return train_img , test_img

        except Exception as e:
            logger.error(f"Error converting image to array and grayscale {img_path}: {e}")
            raise e
            
    def save_images_in_data_ingestion_artifact(self, train_img : any , test_img : any):
        try:
            train_img.save(self.config.output_train_data)
            test_img.save(self.config.output_test_data)
            logger.info(f"image saved at {self.config.output_artifact_path} ")

            
        except Exception as e:
            logger.error(f"Error saving image array to {output_path}: {e}")
            raise e

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            logger.info("<____ Starting data ingestion process ____>")
            
            logger.info("image loading and their extension checking is started....")
            train_data_1 , test_data_1 = self.load_train_and_test_data(image_path = self.config.Data_file_path)
            logger.info("image loading and their extension checking is ended....")

            logger.info("image resizing starts....")
            train_data_2 , test_data_2 = self.normalize_images(train_img = train_data_1 , test_img = test_data_1)
            logger.info("image resizing ends....")

            logger.info("converting image to array and grayscale is starts.....")
            train_data_3 , test_data_3 = self.skip_corrupted_image_and_performance_optimization(train_img = train_data_2 , test_img = test_data_2)
            logger.info("converting image to array and grayscale is ends.....")

            logger.info("image is saving ....")
            self.save_images_in_data_ingestion_artifact(train_img = train_data_3 , test_img = test_data_3)
            logger.info(f"image is saved at {self.config.output_artifact_path} ....")


            return DataIngestionArtifact(
                train_data_path = self.config.output_train_data , 
                test_data_path = self.config.output_test_data
            )



            logger.info("<____ End data ingestion process ____>")
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise e