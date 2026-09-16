import os
from pathlib import Path
import logging

#logging string
# logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'amanClassifier'

list_of_files = [
    ".github/workflows/main.yaml",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/model_building.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/logging/logger.py",
    "k8s/service.yaml",
    "terraform/",
    "test/",
    "fastapi/App/app.py",
    "fastapi/Dockerfile",
    "fastapi/requirements.txt"
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "frontend/templates/index.html",
    "frontend/templates/style.css",
    "backend/App/app.py",
    "backend/Dockerfile",
    "backend/requirements.txt",
    "mlflow/Dockefile",
    "mlflow/requirements.txt",
    "mlflow/server.sh",
    '.Dockerignore',
    'docker-compose.yaml'


]

for fliepath in list_of_files:
    fliepath = Path(fliepath)

    filedir , filename = os.path.split(fliepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok = True)

    if (not os.path.exists(filename)) or (os.path.getsize(fliepath) == 0):
        with open(fliepath , 'w') as f:
            pass
    
    else:
        print("file already exists")
