import cv2
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from imgaug import augmenters
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

data_file = pd.read_csv('data.csv')


def loadData(data):
    imagesPath = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        imagesPath.append(f'data/IMG/{indexed_data[0]}')
        steering.append(float(indexed_data[3]))
    steering = np.asarray(steering)
    return imagesPath, steering


print('Loading the Data')
imgPath, Steering = loadData(data_file)

xtrain, xval, ytrain, yval = train_test_split(imgPath, Steering, test_size=0.2)


def augmentImage(imgPath, steering):
    img = plt.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if np.random.rand() < 0.5:
        pan = augmenters.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = augmenters.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = augmenters.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering


def preProcess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = cv2.imread(imagesPath[index])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                steering = steeringList[index]
            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield np.asarray(imgBatch), np.asarray(steeringBatch)


def createModel():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')
    return model


model = createModel()
model.summary()

model.fit(batchGen(xtrain, ytrain, 100, 1),
          steps_per_epoch=100,
          validation_data=batchGen(xval, yval, 100, 0),
          validation_steps=100, epochs=10)
print('Model Saved')
model.save('model.h5')
