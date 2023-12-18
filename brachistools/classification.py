from keras.models import load_model
import numpy as np
import skimage.transform
import skimage.io
from pathlib import Path
from configparser import ConfigParser
import sys, os

global logger

def classification_pipeline(input_image):
    from configparser import ConfigParser

    try:
        config_path = 'config.ini'
        config = ConfigParser()
        config.read(config_path)
    except:
        logger.critical("Failed to open config file")
        raise RuntimeError("Failed to open config file")

    param_dir = config.get('ModelParams', 'param_dir').strip('\"\'')
    model = load_model(os.path.join(param_dir, 'model.h5'))
    classes = ['benign', 'malignant']

    # In training, `cv2.resize` is used
    # `skimage` resize has slightly different output but does not
    # affect the output significantly
    # Using skimage only would allow us to drop the requirement on
    # opencv-python
    im = skimage.transform.resize(input_image, (224, 224),
        preserve_range=True, anti_aliasing=False).astype(np.uint8)
    predict_results = model.predict(np.expand_dims(im, axis=0), verbose=0)

    predict_class = classes[np.argmax(predict_results)]
    confidence_score = format(np.max(predict_results), '.4f')
    return predict_class, confidence_score
