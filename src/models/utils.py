from src.models.img2name.img2name import Img2Name

def get_model_class(model_key):
    models = {
        "simple_img_name": Img2Name
    }
    loaders = {
        "simple_img_name": "ImgNameLoader"
    }

    model = models.get(model_key, None)
    loader = loaders.get(model_key, None)
    return model, loader