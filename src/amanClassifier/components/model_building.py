from src.amanClassifier.logging.logger import logger
from src.amanClassifier.constants import PARAMS_FILE_PATH
from src.amanClassifier.utils.common import load_yaml
from src.amanClassifier.Artifact.project_artifact import (
    DataIngestionArtifact,
    ModelBuildingArtifact
)
from src.amanClassifier.config.configuration import ModelBuildingConfig

import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2 , VGG16
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)


class ModelBuilding:

    def __init__(
        self,
        config: ModelBuildingConfig,
        artifact: DataIngestionArtifact
    ):

        self.config = config
        self.artifact = artifact

        self.params = load_yaml(
            PARAMS_FILE_PATH
        ).model_building

    def load_pretrained_model(self):

        try:

            logger.info(
                "Loading pretrained MobileNetV2 model..."
            )

            logger.info(
                f"model input shape {self.params.input_shape}"
            )

            conv_base = MobileNetV2(
                weights=self.params.weights,
                include_top=self.params.include_top,
                input_shape=self.params.input_shape
            )

            logger.info(
                "Pretrained MobileNetV2 model loaded successfully."
            )

            conv_base.trainable = False

            logger.info(
                "MobileNetV2 layers are frozen."
            )

            return conv_base

        except Exception as e:

            logger.error(
                f"Error while loading MobileNetV2: {e}"
            )

            raise e

    def build_new_model(self, hp):

        try:

            logger.info(
                "Building model for Keras Tuner..."
            )

            conv_base = self.load_pretrained_model()

            model = Sequential()

            model.add(conv_base)

            model.add(
                GlobalAveragePooling2D()
            )

            num_layers = hp.Int(
                "num_layers",
                min_value=self.params.num_layers_min,
                max_value=self.params.num_layers_max,
                step=self.params.num_layers_step
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
                    min_value=self.params.dropout_min,
                    max_value=self.params.dropout_max,
                    step=self.params.dropout_step
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
                    self.params.num_classes,
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


            logger.info("model compilation start....")
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
                loss="categorical_crossentropy",
                metrics=["accuracy"]
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

                max_trials=self.params.max_trials,

                executions_per_trial=self.params.executions_per_trial,

                directory=self.config.model_building_dir,

                project_name=self.params.project_name,

                overwrite=self.params.overwrite
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

            logger.info(
                "========== MODEL BUILDING STARTED =========="
            )

            tuner = self.make_tuner_and_return()

            logger.info(
                "========== MODEL BUILDING ENDED =========="
            )

            return ModelBuildingArtifact(
                tuner=tuner
            )

        except Exception as e:

            logger.error(
                f"Error during model building: {e}"
            )

            raise e