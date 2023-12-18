from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import cv2
import os
from PIL import Image
from tqdm import tqdm


def load_data(directory, norm_size):
    IMG = []
    read = lambda imagename: np.asarray(Image.open(imagename).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(directory)):
        PATH = os.path.join(directory, IMAGE_NAME)
        _, figuretype = os.path.splitext(PATH)
        if figuretype == ".png":
            img = read(PATH)
            img = cv2.resize(img, (norm_size, norm_size))
            IMG.append(np.array(img))
    return IMG


# Read raw data
benign_test = np.array(load_data('../Data/BreaKHis 400X/test/benign', 224))
malign_test = np.array(load_data('../Data/BreaKHis 400X/test/malignant', 224))

# Label
benign_test_label = np.zeros(len(benign_test))
malign_test_label = np.ones(len(malign_test))

# Merge data for model testing. X is the data while Y is the label.
X_test = np.concatenate((benign_test, malign_test), axis=0)
Y_test = np.concatenate((benign_test_label, malign_test_label), axis=0)

# Shuffle test data
s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

# One-hot
Y_test = to_categorical(Y_test, num_classes=2)

model = load_model('model.h5')

# Evaluation classification results
score = model.evaluate(X_test, Y_test, verbose=1)
print(score)
