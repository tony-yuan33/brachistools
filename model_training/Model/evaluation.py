
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, auc, roc_curve
from tqdm.contrib import itertools


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

Y_pred = model.predict(X_test)
report = classification_report(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
print(report)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


cm = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))

cm_plot_label = ['benign', 'malignant']
plot_confusion_matrix(cm, cm_plot_label, title='Confusion Metrix')
plt.show()

roc_log = roc_auc_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
false_positive_rate, true_positive_rate, threshold = roc_curve(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
area_under_curve = auc(false_positive_rate, true_positive_rate)

plt.plot([0, 1], [0, 1], 'r--')
plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
plt.close()