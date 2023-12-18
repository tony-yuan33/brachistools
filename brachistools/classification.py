from keras.models import load_model
import numpy as np
import cv2
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
    input_image = cv2.resize(input_image, (224, 224))
    img = np.array(input_image)
    predict_results = model.predict(np.expand_dims(img, axis=0), verbose=0)
    predict_class = classes[np.argmax(predict_results)]
    confidence_score = format(np.max(predict_results), '.4f')
    return predict_class, confidence_score
