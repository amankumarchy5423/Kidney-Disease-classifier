from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import Annotated

class UserInput(BaseModel):
    image: Annotated[
        UploadFile,
        File(description="Add an image of kidney")
    ]