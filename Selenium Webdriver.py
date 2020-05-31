
import cv2 
import imutils
import numpy as np 
import pickle 
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time 
import tensorflow.keras.backend
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense

#this is just a test for the driver, because it generated a Chrome version conflict

driver = webdriver.Chrome(ChromeDriverManager().install()) 
driver.get("https://www.google.com/?hl=ro")
#finding the element from the page
search_bar = driver.find_element_by_name("q") 
#send some text 
search_bar.send_keys("facebook")
search_bar.submit()
time.sleep(1) 
driver.close()


#add all I previously did to a pipeline (image pre-processing & normalization)
#load the model so it doesnt have to load during runtime to predict 

def read_img(image): 
    image_to_array = cv2.imread(image)
    return image_to_array

def grayscale(image): 
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return gray

def threshold(grayscaled_img): 
    thresholded = cv2.threshold(grayscaled_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    return thresholded

def dilate(thresholded_img): 
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(thresholded_img, kernel, iterations = 1) 
    return dilated

def get_contours(thresholded_img): 
    contours = cv2.findContours(thresholded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours

def get_ROI(contours): 
    boundingRectangles = map(cv2.boundingRect,contours)
    return list(boundingRectangles)

def split_chars(rectangles): 
    splitted = []
    for r in rectangles: 
        x,y,w,h = r
        if w/h > 1.25: 
            new_w = int(w/2) 
            splitted.append((x, y, new_w, h))
            splitted.append((x+new_w, y, new_w, h))
        else: 
            splitted.append(r)
    return splitted

def extract_single_char(image, rectangles): 
    single_char = [] 
    for r in rectangles: 
        x,y,w,h = r
        single_char_img = image[y : y+h, x : x+w] 
        single_char.append(single_char_img) 
    return single_char

def sort_ROI(rectangles): 
    sorted_br = sorted(rectangles, key = lambda x: float(x[0]))
    return sorted_br

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

def resize_to_keras(normalized_image): 
    to_keras = np.expand_dims(normalized_image, axis=2)
    return to_keras

#perform all the steps from preprocessing in a new method to get a single char from captcha img
def image_preprocessing_pipeline(captcha_image): 
    captcha_img = read_img(captcha)
    grayscaled_img = grayscale(captcha_img)
    thresholded_img = threshold(grayscaled_img)
    dilated_img = dilate(thresholded_img)
    contours = get_contours(dilated_img)
    bounding_rectangles = get_ROI(contours)
    splitted_chars = split_chars(bounding_rectangles)
    splitted_chars = sort_ROI(splitted_chars)
    single_chars = extract_single_choptions = webdriver.ChromeOptions()
    return single_chars

def load_model_weights(): 
    nr_classes = 32
    my_model = Sequential() 
    my_model.add(Conv2D(20, (5,5), padding = "same", input_shape=(20,20,1), activation = "relu" )) 
    my_model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    my_model.add(Conv2D(50, (5,5), padding = "same", activation = "relu")) 
    my_model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2))) 
    my_model.add(Flatten()) 
    my_model.add(Dense(512, activation = "relu")) 
    my_model.add(Dense(nr_classes, activation = "softmax"))
    
    loaded_model = my_model.load_weights("myModelWeights.h5")
    return loaded_model
    
#load the label binarizer for classes   
def load_LB(): 
    lb = pickle.load(open("my_lb.pickle","rb")) 
    return lb
    
#put them together in a single method   
def load_model_and_lb(): 
    my_model = load_model_weights() 
    lb = load_LB()
    return my_model, lb 
    
def predict(single_chars, model, lb): 
    X = [] 
    for single_char_img in single_chars: 
        grayscaled = grayscale(single_char_img) 
        normalized = normalize_img(single_char_img)
        keras_img = resize_to_keras(normalized) 
    X.append(keras_img) 
    #binarize the keras imgs 
    X = np.array(X, dtype = "float")/255.0 
    prediction = model.predict(X) 
    #Transform binary labels back to multi-class labels
    predicted_char = lb.inverse_transform(prediction)
    return predicted_char 
    
#get the previously calculated model weights  
model, lb = load_model_and_lb()

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import urllib.request 

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(ChromeDriverManager().install(),options=options) 

driver.get("http://mycaptchagenerator.c1.biz/captcha-if-you-can/") 
time.sleep(2)

captcha_form = driver.find_element_by_css_selector("img.wpcf7-form-control") 

opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)
url = captcha_form.get_attribute("src") # get the image source
local = "captcha.png"
urllib.request.urlretrieve(url, local) #download the image with the name captcha.png

predicted_captcha = predict(image_preprocessing_pipeline("captcha.png"), model, lb)

submix_box = driver.find_element_by_name("captcha-170")
submit_box.send_keys(pred)
time.sleep(10)
driver.quit()


