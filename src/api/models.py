from pydantic import BaseModel

class InferenceRequest(BaseModel):
    imageSrc: str
    diversity: float
    min_name_length: int