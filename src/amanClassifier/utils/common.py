import os
import yaml
from src.amanClassifier.logger import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


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


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())


