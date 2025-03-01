from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    testing: bool = False

settings = Settings()

IMG2NAME_WEIGHTS_PATH = "/app/src/models/img2name/files/weights.pt"
IMG2NAME_MAPS_PATH = "/app/src/models/img2name/files/maps.pkl"
IMG2NAME_PARAMETERS_PATH = "/app/src/models/img2name/files/params.pkl"
