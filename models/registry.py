MODEL_REGISTRY = dict()

def register_model(model_class):

    MODEL_REGISTRY[model_class.model_name] = model_class

    return model_class