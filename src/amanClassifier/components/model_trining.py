from src.amanClassifier.logging.logger import logger
from src.amanClassifier.constants import PARAMS_FILE_PATH
from src.amanClassifier.utils.common import read_yaml,save_binary_file , load_image_data,extract_x_y
from src.amanClassifier.Artifact.project_artifact import ( ModelBuildingArtifactArtifact,ModelTrainingArtifact,DataIngestionArtifact )
from src.amanClassifier.config.configuration import ModelBuildingConfig


import os
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping


class ModelTraining:
    def __init__(self,config:ModelTrainingConfig, model_artifact:ModelTrainingArtifact, data_artifact:DataIngestionArtifact):
        self.config = config
        self.model_artifact = artifact
        self.data_artifact = data_artifact
        self.params = read_yaml(PARAMS_FILE_PATH).model_training
    
    def select_best_model(self,x_train : any ,y_train : any ,x_val : any ,y_val : any):
        try:
            early_stopping = EarlyStopping(
                monitor=self.params.earlystopping_monitor,
                patience=5,
                restore_best_weights=True
            )
            logger.info("EarlyStopping callback created.")

            logger.info("Starting Keras Tuner search...")
            tuner = self.model_artifact.tuner
            tuner.search(
                x_train,
                y_train,
                validation_data=(x_val , y_val),
                epochs=self.params.tunner_epochs,
                callbacks=[early_stopping]
            )
            logger.info("Keras Tuner search completed.")
            logger.info(f"best hyperparameter of model {
                tuner.get_best_hyperparameters()
            }")

            best_model = tuner.get_best_models(
                num_models = 1
            )[0]
            logger.info("got best model from tuner ....")

            return best_model,early_stopping

        except Exception as e:
            logger.error(e)
            raise e

    def models_parameter_and_structure(self,best_model:any):
        try:
            model_summary = model.summary()
            logger.info(f"model summary {model_summary}")

            plot_model(
                best_model,
                to_file = self.config.model_png,
                show_shapes = True,
                show_layer_names=True,
                dpi = 100,
                show_layer_activations=True,
                show_trainable = True
            )

            logger.info(f"model info plateed at {self.config.model_png}")
        except Exception as e:
            logger.error(e)
            raise e

    def model_training(self,
                        best_model:any,
                        early_stop:any,
                        x_train : any,
                        y_train : any,
                        x_val : any ,
                        y_val : any,
                        x_test : any,
                        y_test : any
                        ):
        try:
            logger.info("model training starts....")
            best_model.fit(
                x_train,
                y_train,
                epoch = self.params.model_train_epoch,
                initial_epochs = self.params.tunner_epochs+1,
                validation_data = (x_val , y_val),
                batch_size = self.params.training_batch_size,
                verbose = 1 ,
                callbacks = [early_stop]
            )

            loss , accuracy = best_model(
                x_test , 
                y_test
            )
            logger.info("loss of model : {loss} , accuracy of model {accuracy}")

            save_binary_file(content = best_model , file_path = self.config.model_file_path)

        except Exception as e:
            logger.error(e)
            raise e

    def initiate_model_training(self):
        try:
            logger.info("_______ MODEL TRAINING STARTED ______")

            train_data = load_image_data(self.data_artifact.train_data_path)
            x_train , y_train = extract_x_y(train_data)
            logger.info(f"train data is loaded and x_train : \n {x_train} and y_train \n {y_train}")

            test_data = load_image_data(self.data_artifact.test_data_path)
            x_test , y_test = extract_x_y(test_data)
            logger.info(f"train data is loaded and x_train : \n {x_test} and y_train \n {y_test}")

            val_data = load_image_data(self.data_artifact.val_data_path)
            x_val , y_val = extract_x_y(val_data)
            logger.info(f"train data is loaded and x_train : \n {x_val} and y_train \n {y_val}")

            best_model , early_stop = self.select_best_model(
                x_train = x_train,
                y_train = y_train,
                x_val = x_val,
                y_val = y_val
            )

            self.models_parameter_and_structure(best_model=best_model)

            self.model_training(
                best_model = best_model,
                early_stop = early_stop,
                x_train = x_train ,
                y_train = y_train,
                x_val = x_val ,
                y_val = y_val,
                x_test = x_test,
                y_test = y_test
            )
            


            logger.info("_______ MODEL TRAINING ENDED ______")
        except Exception as e:
            logger.error(e)
            raise e



    