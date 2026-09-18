from src.amanClassifier.logging.logger import logger
from src.amanClassifier.components.data_ingestion import DataIngestion
import tensorflow as tf
import pytest



def test_split_test_into_val_and_test():

    data_ingestion = DataIngestion.__new__(DataIngestion)

    images = tf.random.uniform(
        (100, 224, 224, 3)
    )

    labels = tf.one_hot(
        tf.random.uniform(
            (100,),
            maxval=4,
            dtype=tf.int32
        ),
        depth=4
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (images, labels)
    ).batch(10)

    val_data, test_data = (
        data_ingestion.split_test_into_val_and_test(dataset)
    )

    val_batches = tf.data.experimental.cardinality(
        val_data
    ).numpy()

    test_batches = tf.data.experimental.cardinality(
        test_data
    ).numpy()

    assert val_batches + test_batches == 10