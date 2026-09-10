from src.amanClassifier.logging.logger import logger
from src.amanClassifier.constants import *
from src.amanClassifier.utils.common import read_yaml
from src.amanClassifier.Artifact.project_artifact import (
    DataIngestionArtifact,
    ModelBuildingArtifact
)
from src.amanClassifier.config.configuration import ModelBuildingConfig


import os
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import EarlyStopping



class ModelBuilding:

    def __init__(
        self,
        config: ModelBuildingConfig,
        artifact: DataIngestionArtifact
    ):

        self.config = config
        self.artifact = artifact

        self.params = read_yaml(
            PARAMS_FILE_PATH
        ).model_building

    def load_pretrained_vgg16_model(self):

        try:

            logger.info(
                "Loading pretrained VGG16 model..."
            )

            conv_base = VGG16(
                weights=self.params.weights,
                include_top=self.params.include_top,
                input_shape=self.params.input_shape
            )

            logger.info(
                "Pretrained VGG16 model loaded successfully."
            )

            conv_base.trainable = False

            logger.info(
                "VGG16 layers are frozen."
            )

            return conv_base

        except Exception as e:

            logger.error(
                f"Error while loading VGG16: {e}"
            )

            raise e

    def build_new_model(self, hp):

        try:

            logger.info(
                "Building model for Keras Tuner..."
            )


            conv_base = self.load_pretrained_vgg16_model()

            model = Sequential()

            model.add(conv_base)

            model.add(
                GlobalAveragePooling2D()
            )

            num_layers = hp.Int(
                "num_layers",
                min_value=1,
                max_value=3,
                step=1
            )

            logger.info(
                f"Number of Dense layers selected: {num_layers}"
            )

            for i in range(num_layers):

                units = hp.Choice(
                    f"units_{i}",
                    values=self.params.num_of_nodes
                )

                dropout_rate = hp.Float(
                    f"dropout_{i}",
                    min_value=0.2,
                    max_value=0.5,
                    step=0.1
                )

                model.add(
                    Dense(
                        units=units,
                        activation="relu"
                    )
                )

                model.add(
                    Dropout(
                        rate=dropout_rate
                    )
                )

            model.add(
                Dense(
                    4,
                    activation="softmax"
                )
            )

            optimizer_name = hp.Choice(
                "optimizer",
                values=self.params.optimizers
            )

            learning_rate = hp.Choice(
                "learning_rate",
                values=self.params.learning_rate
            )

            if optimizer_name == "adam":

                optimizer = keras.optimizers.Adam(
                    learning_rate=learning_rate
                )

            elif optimizer_name == "nadam":

                optimizer = keras.optimizers.Nadam(
                    learning_rate=learning_rate
                )

            else:

                optimizer = keras.optimizers.RMSprop(
                    learning_rate=learning_rate
                )

            model.compile(
                optimizer=optimizer,

                loss=self.params.loss_function,

                metrics=[
                    "accuracy"
                ]
            )

            logger.info(
                "Model compiled successfully."
            )

            return model

        except Exception as e:

            logger.error(
                f"Error while building model: {e}"
            )

            raise e

    def make_tuner_and_return(self):

        try:

            logger.info(
                "Initializing Keras Tuner..."
            )

            tuner = kt.RandomSearch(

                hypermodel=self.build_new_model,

                objective=kt.Objective(
                    "val_accuracy",
                    direction="max"
                ),

                max_trials=10,

                executions_per_trial=1,

                directory=self.config.model_building_dir,

                project_name="amanClassifier",

                overwrite=True
            )

            logger.info(
                "Keras Tuner initialized successfully."
            )

            return tuner

        except Exception as e:

            logger.error(
                f"Error while creating Keras Tuner: {e}"
            )

            raise e

    def initiate_model_building(self):

        try:

            logger.info("========== MODEL BUILDING STARTED ==========" )

            tuner = (
                self.make_tuner_and_return()
            )

            logger.info("========== MODEL BUILDING EndED ==========" )

            return ModelBuildingArtifact(
                tuner=tuner
            )

        except Exception as e:

            logger.error(
                f"Error during model building: {e}"
            )

            raise e