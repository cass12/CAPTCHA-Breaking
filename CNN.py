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
from sklearn.model_selection import train_test_split 
import pickle


def read_img(image): 
    image_to_array = cv2.imread(image)
    return image_to_array

def display_img(image):
    cv2.imshow('Characters', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    
def grayscale(image): 
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return gray

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
    return img_border_resized

#keras requires 4 dims 
def resize_to_keras(normalized_image): 
    to_keras = np.expand_dims(normalized_image, axis=2)
    return to_keras

imgs = [] 
labels = []
extracted_characters_folder = "/home/cas/Desktop/Licenta_App/extracted_chars1" 
for character_image in paths.list_images(extracted_characters_folder): 
    single_character = read_img(character_image) 
    grayscaled = grayscale(single_character) 
    normalized = normalize_img(grayscaled) 
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
my_model = Sequential() 
my_model.add(Conv2D(20, (3,3), padding = "same", input_shape=(20,20,1), activation = "relu" )) 
my_model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
my_model.add(Conv2D(50, (3,3), padding = "same", activation = "relu")) 
my_model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2))) 
my_model.add(Flatten()) 
my_model.add(Dense(384, activation = "relu")) 
my_model.add(Dense(nr_classes, activation = "softmax"))

my_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
my_model.summary()

#A nice way of displaying the architecture
from ann_visualizer.visualize import ann_viz 
ann_viz(my_model, title="The CNN Architecture")

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle=True)
my_model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=11)


#serializing model and label binarizer
my_model.save_weights("myModelWeights.h5") 
f = open("my_lb.pickle","wb")
f.write(pickle.dumps(lb))
f.close()
