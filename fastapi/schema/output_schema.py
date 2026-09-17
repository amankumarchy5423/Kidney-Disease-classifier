from pydantic import BaseModel , Field
from typing import Optional , Literal


class OutputSchema(BaseModel) :
    output_class : Literal["Cyst","Normal","Stone","Tumor"] = Field(...,description="condition of kidney")
    confidence : float = Field(...,description="probability of the output condition of kidney",gt=0.5)