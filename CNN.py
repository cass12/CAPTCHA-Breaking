### Work in progress ###

import numpy as np
import cv2
import os
import imutils
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import backend
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MAXPooling2D
from keras.layers.core import Flatten, Dense 

def read_img(image):
    return cv2.imread(character_image)

def display_img(image):
    cv2.imshow('Characters', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows

def normalize_img(image, new_width=20, new_height=20):
    #define image dims
    #slicing, start:stop-1
    (h, w) = image.shape[:2]
    if w > h:
        image = imutils.resize(image, width = new_width)
    else:
        image = imutils.resize(image, height = new_height)
    #define size of padding area
    h_padding = int((new_height - image.shape[0])/2)
    w_padding = int((new_width - image.shape[1])/2)
    white = [255,255,255]
    img_with_border = cv2.copyMakeBorder(image, h_padding, h_padding, w_padding, w_padding, cv2.BORDER_CONSTANT, value = white)
    img_border_resized = cv2.resize (image, (new_width, new_height), interpolation = cv2.INTER_AREA)
    return img_with_border

#keras requires 4 dims
def resize_to_keras(image):
    return np.expand_dims(image, axis=2)

imgs = []
labels = []
extracted_characters_folder = "/home/cas/Desktop/extracted_characters"
for character_image in paths.list_images(extracted_characters_folder):
    print(character_image)
    single_character = read_img(extracted_characters_folder)
    #display_img(single_character)
    grayscaled = grayscale_img(single_character)
    normalized = normalize_img(grayscaled)
    #display_img(normalized)
    keras_img = resize_to_keras(normalized)
    imgs.append(keras_img)
    label = character_image.split(os.path.sep)[-2]
    labels.append(label)


bw_to_binary = np.array(imgs, dtype=object)/255.0 #binarizing the colours
labels = np.array(labels)

lb = LabelBinarizer().fit(labels) #training on the labels
binary_class_encoder = lb.transform(labels) #encodes each label into a vector for its specific class 

#A simple CNN Sequential architecture with one in/out tensor
nr_classes = len(set(labels))
my_model = Sequential 
model.add(Conv2D(20, (3,3), padding = "same", input(20,20,1), activation = "relu" )) 
model.add(MAXPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Conv2D(50, (3,3), padding = "same", input(20,20,1), activation = "relu")) 
model.add(MAXPooling2D(pool_size = (2,2), strides = (2,2))) 
model.add(Flatten()) 
model.add(Dense(384, activation = "relu")) 
model.add(Dense(nr_classes, activation = "softmax"))
