
import os

import keras
import torch
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.src.utils import load_img, to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from torch import nn
from torch.nn import BatchNorm1d
from tqdm.notebook import tqdm
from keras.src.callbacks import EarlyStopping
import tensorflow as tf

earlystopping = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=2)


def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, target_size=(236, 236))
        #img = tf.image.random_crop(img, size=(224, 224, 3))
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(features.shape[0], 236, 236, 3)  # Reshape all images in one go
    return features

TRAIN_DIR = "C:/Users/aryam/OneDrive/Desktop/Data/Train"

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

train_features = extract_features(train['image'])

x_train = train_features / 255.0

le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_train = to_categorical(y_train, num_classes=2)

model = Sequential()
# Convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(236, 236, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


#model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(1024, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dense(2, activation='softmax'))
#lr_schedule = keras.optimizers.schedules.ExponentialDecay(0.001,decay_steps=60,decay_rate=0.5,staircase=True)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x=x_train, y=y_train, batch_size=24, epochs=10,shuffle=True, callbacks=[earlystopping])

test_image_dir='C:/Users/aryam/OneDrive/Desktop/Data/Test'

import imghdr

def create_test_dataframe(dir):
    image_paths = []
    image_names = []
    for filename in os.listdir(dir):
      file_path = os.path.join(dir, filename)
      if imghdr.what(file_path) is not None:
        #if filename.endswith('.jpg'):
         image_paths.append(file_path)
         image_names.append(filename)

    return image_paths,image_names



label_mapping = {0: 'AI', 1: 'Real'}

test=pd.DataFrame()
test['image'],test['image_name']=create_test_dataframe(test_image_dir)
print(test['image_name'])
test_features=extract_features(test['image'])
x_test=test_features/255.0

predictions=model.predict(x_test)
predicted_labels=np.argmax(predictions,axis=1)

results=pd.DataFrame({'Id': test['image_name'], 'label':predicted_labels})
results['label']=results['label'].map(label_mapping)
results.to_csv('ary_predictions.csv',header=True, index=False)