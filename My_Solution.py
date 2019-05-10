#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 19:58:12 2019

@author: srinivas
"""

#Importing packages
import numpy as np
import pandas as pd
from PIL import Image
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from glob import glob
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2

IMAGE_SIZE = [224, 224] 
epochs = 5
batch_size = 32
train_path = '/home/srinivas/Documents/Hackatons/Jio_Artificial_Intelligence/My_model/dataset/A'
valid_path = '/home/srinivas/Documents/Hackatons/Jio_Artificial_Intelligence/My_model/dataset/B'
 #useful for getting number of files
image_files = glob(train_path + '/*/*.jpg')
valid_image_files = glob(valid_path + '/*/*.jpg')
folders = glob(train_path + '/*')

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
  layer.trainable = False
  
# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='sigmoid')(x)


# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='binary_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)

# create an instance of ImageDataGenerator
gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)

# get label mapping for confusion matrix plot later
test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
  labels[v] = k

# should be a strangely colored image (due to VGG weights being BGR)
for x, y in test_gen:
  print("min:", x[0].min(), "max:", x[0].max())
  plt.title(labels[np.argmax(y[0])])
  plt.imshow(x[0])
  plt.show()
  break


# create generators
train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)


# fit the model
r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
)

def get_confusion_matrix(data_path, N):
  # we need to see the data in the same order
  # for both predictions and targets
  print("Generating confusion matrix", N)
  predictions = []
  targets = []
  i = 0
  for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):
    i += 1
    if i % 50 == 0:
      print(i)
    p = model.predict(x)
    p = np.argmax(p, axis=1)
    y = np.argmax(y, axis=1)
    predictions = np.concatenate((predictions, p))
    targets = np.concatenate((targets, y))
    if len(targets) >= N:
      break

  cm = confusion_matrix(targets, predictions)
  return cm


cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)


# plot some data

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()

from util import plot_confusion_matrix
plot_confusion_matrix(cm, labels, title='Train confusion matrix')
plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix')


# Loading the CSV file which contains the details of the parking lot in the camera
positions = pd.read_csv('camera5.csv')
p = np.array(positions)

result ={}
for i in range(0,len(p)):
    a = p[i]
    lot_id = a[0]
    x1 = a[1]
    y1 = a[2]
    w = a[3]
    h = a[4]
    # The given points are extracted from a 2592*1944 image and the given image for is 1000/750
    # So we rescale our points
    x1 = x1*1000/2592
    y1 = y1*750/1944
    w = w*1000/2592
    h = h*750/1944
    x2 = x1+w
    y2 = y1+h
    im = Image.open("input.jpg")
    crop_rectangle = (x1, y1, x2, y2)
    cropped_im = im.crop(crop_rectangle)
    cropped_im.save('out.jpg')
    # load an image from file
    image = load_img('out.jpg', target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # prepare the image for the VGG model
    image = preprocess_input(image)
    y_pred = model.predict(image)
    if y_pred> 0.5:
        result[lot_id] = 1
    else:
        result[lot_id] = 0
        
for i in range(0,len(p)):
    a = p[i]
    lot_id = a[0]
    x1 = a[1]
    y1 = a[2]
    w = a[3]
    h = a[4]
    x1 = x1*1000/2592
    y1 = y1*750/1944
    w = w*1000/2592
    h = h*750/1944
    x2 = x1+w
    y2 = y1+h
    if result[lot_id] ==0:
        img = cv2.imread('input.jpg',cv2.IMREAD_COLOR)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
        cv2.imwrite('input.jpg', img)
    else:
        img = cv2.imread('input.jpg',cv2.IMREAD_COLOR)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.imwrite('input.jpg', img)

print("The processingi is done: please refer input.jpg")
