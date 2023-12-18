from keras.models import load_model
import numpy as np
import cv2


def classification_pipeline(input_image):
    model = load_model('brachistools/models/model.h5')
    classes = ['benign', 'malignant']
    input_image = cv2.resize(input_image, (224, 224))
    img = np.array(input_image)
    predict_results = model.predict(np.expand_dims(img, axis=0), verbose=0)
    predict_class = classes[np.argmax(predict_results)]
    confidence_score = format(np.max(predict_results), '.4f')
    return predict_class, confidence_score
