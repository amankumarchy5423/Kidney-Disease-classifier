import os
import yaml
from src.amanClassifier.logging.logger import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import tensorflow as tf
import numpy as np


@ensure_annotations
def extract_x_y(dataset):
    try:
        x_list, y_list = [], []
        for images, labels in dataset:
            x_list.append(images.numpy())
            y_list.append(labels.numpy())

        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return x, y
    except Exception as e:
        logger.error(f"Error extracting x, y from dataset: {e}")
        raise e

# dump model 
@ensure_annotations
def save_binary_file(content : any , file_path : Path):

    joblib.dump(content,filename=file_path)
    logger.info(f"binary file saved at{file_path}")



@ensure_annotations
def load_binary_file(file_path:Path):

    bin_file =  joblib.load(filename=file_path)
    logger.info(f"binary file loaded from location {file_path}")
    return bin_file


@ensure_annotations
def create_directories(file_path_list:list,verbose = True):
    for file in file_path_list:
        os.makedirs(file,exist_ok=True)
        if verbose:
            logger.info(f" this  {file} created")


@ensure_annotations
def save_in_json(content:dict , file_path : Path):
    try:
        with open(file_path , 'w') as f :
            json.dump(content ,f,indent=4)
    
        logger.info(f'content saved at {file_path}')
    except Exception as e :
        logger.error(e)
        raise e


@ensure_annotations
def load_from_json(file_path:Path):
    try:
        with open(file_path) as f :
            content = json.load(f)
        
        logger.info(f"content loaded from file : {file_path}")
        return ConfigBox(content)
    except Exception as e:
        logger.error(e)
        raise e


@ensure_annotations
def load_yaml(file_path:Path):
    try:
        with open(file_path) as f :
            content = yaml.safe_load(f)
        logger.info("content is loaded ")
        return ConfigBox(content)
    except Exception as e:
        logger.error(e)
        raise e
    

@ensure_annotations
def save_yaml(content:dict , file_path:Path):
    try:
        
        yaml.safe_dump(content,file_path)
        logger.info(f"content dump safely at {file_path}")

    except Exception as e :
        logger.error(e)
        raise e 

@ensure_annotations
def save_image_data(img_data, filename):
    try:
        dir_name = os.path.dirname(filename)
        create_directories([dir_name])
        img_data.save(filename)
    except Exception as e:
        logger.error(e)
        raise e


def load_image_data(file_path):
    try : 
        Data = tf.data.Dataset.load(file_path)
    except Exception as e :
        logger.error(e)
        raise e


