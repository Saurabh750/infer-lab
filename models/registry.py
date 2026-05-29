MODEL_REGISTRY = dict()

def register_model(model_class):

    for model_name in model_class.model_names:
        MODEL_REGISTRY[model_name] = model_class

    return model_class