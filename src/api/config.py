from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    testing: bool = False

settings = Settings()

WEIGHTS_PATH = "/app/src/models/img2name/files/weights.pt"
MAPS_PATH = "/app/src/models/img2name/files/maps.pkl"
PARAMETERS_PATH = "/app/src/models/img2name/files/params.pkl"
