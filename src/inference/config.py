from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    testing: bool = False

settings = Settings()

MODEL_PATH = "/app/models/img2name/files/img2name.keras"
MAPS_PATH = "/app/models/img2name/files/maps.pkl"
