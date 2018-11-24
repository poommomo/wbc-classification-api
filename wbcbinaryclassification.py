from os import listdir
import cv2
from cv2 import imread
from scipy.misc import imresize
from numpy import asarray, rint
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
import keras.backend as kb
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda

path = './images'


def getWhiteBloodCellTypeList(x):
    return listdir(f'./images/{x}')


def getWhiteBloodCellList(x, type):
    return listdir(f'./images/{x}/{type}')


def getImage(x, type, fileName):
    return imread(f'./images/{x}/{type}/{fileName}')


def getResizeImage(image, size):
    return imresize(image, size)


def toArray(image):
    return asarray(image)


def getData(s):
    x = []
    y = []

    bloodCellType = getWhiteBloodCellTypeList(s)

    eosinophilList = getWhiteBloodCellList(s, bloodCellType[0])
    lymphocyteList = getWhiteBloodCellList(s, bloodCellType[1])
    monocyteList = getWhiteBloodCellList(s, bloodCellType[2])
    neutrophilList = getWhiteBloodCellList(s, bloodCellType[3])

    classes = ['POLYNUCLEAR', 'MONONUCLEAR']

    for i in eosinophilList:
        image = getImage(s, bloodCellType[0], i)
        if image is not None:
            image = getResizeImage(image, (120, 160, 3))
            x.append(toArray(image))
            y.append(classes[0])
    print("eosinophil Done!!")
    for i in lymphocyteList:
        image = getImage(s, bloodCellType[1], i)
        if image is not None:
            image = getResizeImage(image, (120, 160, 3))
            x.append(toArray(image))
            y.append(classes[1])
    print("lymphocyte Done!!")
    for i in monocyteList:
        image = getImage(s, bloodCellType[2], i)
        if image is not None:
            image = getResizeImage(image, (120, 160, 3))
            x.append(toArray(image))
            y.append(classes[1])
    print("monocyte Done!!")
    for i in neutrophilList:
        image = getImage(s, bloodCellType[3], i)
        if image is not None:
            image = getResizeImage(image, (120, 160, 3))
            x.append(toArray(image))
            y.append(classes[0])
    print("neutrophil Done!!")
    x = toArray(x)
    y = toArray(y)
    return x, y


def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x * 1./255.,
                     input_shape=(120, 160, 3), output_shape=(120, 160, 3)))
    model.add(Conv2D(32, (3, 3), input_shape=(120, 160, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    return model


def train_data():
    model = get_model()
    x_train, y_train = getData('TRAIN')
    x_test, y_test = getData('TEST_SIMPLE')

    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    model.fit(x_train, y_train, validation_split=0.2,
              epochs=20, shuffle=True, batch_size=32)
    model.save_weights('binary_model.h5')


def predict_data(x_test):
    model = get_model()
    model.load_weights('binary_model.h5')
    predict = model.predict(x_test)
    kb.clear_session()
    return predict[0][0]
  
