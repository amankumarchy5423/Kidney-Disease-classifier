from src.amanClassifier.logging.logger import logger
from src.amanClassifier.constants import PARAMS_FILE_PATH
from src.amanClassifier.utils.common import (
    load_yaml,
    save_binary_file
)
from src.amanClassifier.Artifact.project_artifact import (
    ModelBuildingArtifact,
    ModelTrainingArtifact,
    DataIngestionArtifact
)
from src.amanClassifier.config.configuration import ModelTrainingConfig

import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import plot_model


class ModelTraining:

    def __init__(
        self,
        config: ModelTrainingConfig,
        model_artifact: ModelBuildingArtifact,
        data_artifact: DataIngestionArtifact
    ):

        self.config = config
        self.model_artifact = model_artifact
        self.data_artifact = data_artifact

        self.params = load_yaml(
            PARAMS_FILE_PATH
        ).model_training

    def select_best_model(
        self,
        train_data,
        val_data
    ):

        try:

            early_stopping = EarlyStopping(
                monitor=self.params.earlystopping_monitor,
                patience=self.params.patience,
                restore_best_weights=True
            )

            logger.info(
                "EarlyStopping callback created."
            )

            logger.info(
                "Starting Keras Tuner search..."
            )

            tuner = self.model_artifact.tuner

            tuner.search(
                train_data,
                validation_data=val_data,
                epochs=self.params.tunner_epochs,
                callbacks=[early_stopping]
            )

            logger.info(
                "Keras Tuner search completed."
            )

            logger.info(
                f"best hyperparameters: "
                f"{tuner.get_best_hyperparameters(num_trials=1)[0].values}"
            )

            best_model = tuner.get_best_models(
                num_models=1
            )[0]

            logger.info(
                "got best model from tuner...."
            )

            return best_model, early_stopping

        except Exception as e:

            logger.error(e)
            raise e

    def models_parameter_and_structure(
        self,
        best_model
    ):

        try:

            best_model.summary()

            # plot_model(
            #     best_model,
            #     to_file=self.config.model_png,
            #     show_shapes=True,
            #     show_layer_names=True,
            #     dpi=self.params.model_plot_dpi,
            #     show_trainable=True
            # )

            logger.info(
                f"model info plotted at {self.config.model_png}"
            )

        except Exception as e:

            logger.error(e)
            raise e

    def model_training(
        self,
        best_model,
        early_stop,
        train_data,
        val_data,
        test_data
    ):

        try:

            logger.info(
                "model training starts...."
            )

            best_model.fit(
                train_data,
                epochs=5,
                validation_data=val_data,
                verbose=1,
                callbacks=[early_stop]
            )
            logger.info("model trained sucessfully.....")

            # loss, accuracy = best_model.evaluate(
            #     test_data,
            #     verbose=1
            # )

            # logger.info(
            #     f"loss of model: {loss:.4f}, "
            #     f"accuracy of model: {accuracy:.4f}"
            # )

            best_model.save('artifact/model_training/model.keras')

            logger.info(
                "model saved successfully."
            )

        except Exception as e:

            logger.error(e)
            raise e

    def initiate_model_training(self):

        try:

            logger.info(
                "_______ MODEL TRAINING STARTED ______"
            )

            train_data = self.data_artifact.train_data
            val_data = self.data_artifact.val_data
            test_data = self.data_artifact.test_data

            best_model, early_stop = self.select_best_model(
                train_data=train_data,
                val_data=val_data
            )

            self.models_parameter_and_structure(
                best_model=best_model
            )

            self.model_training(
                best_model=best_model,
                early_stop=early_stop,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data
            )

            logger.info(
                "_______ MODEL TRAINING ENDED ______"
            )

        except Exception as e:

            logger.error(e)
            raise e