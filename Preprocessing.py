# Preprocessing the images using OpenCV2
# Extracting single character images from a 4 characters CAPTCHA image

import os
import cv2
import numpy as np

#joing paths to get the image basename later on
#*it has a list format to be able to iterate trough all the labels
captcha_folder = "/home/cas/Desktop/Licenta_App/captcha_imgs" 
captcha_dataset = [os.path.join(captcha_folder, f) for f in os.listdir(captcha_folder)]
print (captcha_dataset)


#extract the basename and remove extension to get the labels
def get_label(image):
    filename = os.path.basename(image)
    label = filename.split(".")[0]
    return label


#image to np array
def read_img(captcha_single_img):
    return cv2.imread(captcha_single_img) 

def display(image): 
    cv2.imshow("CAPTCHA", image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grayscale(captcha_img): 
    return cv2.cvtColor(captcha_img, cv2.COLOR_BGR2GRAY)


#returns retVal (Otsu optimal threshold) as first value
#so we take the second output which is the thresholded image 
def threshold(grayscaled_img): 
    return cv2.threshold(grayscaled_img, 0 ,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

#convolves a kernel with the image in order to increase the white region of an image
def dilate(thresholded_img):
    kernel = np.ones((2,2), np.uint8)
    return cv2.dilate(thresholded_img, kernel, iterations = 1)


#returns img, contours & hierarchy
#only the modified image is needed ([0])
def findContours(thresholded_img): 
    return cv2.findContours(thresholded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


#maps the function for each array(contour) to determine the contour for each letter
def calculateBoundingRectangles(contours): 
    return list(map(cv2.boundingRect, contours))


#rectangles are actually the bounding rectangles from countour
def displayBoundingRectangles(rectangles,image):
    for r in rectangles: 
        x,y,w,h=r 
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0), 0) 
    display(image)

def splitChars(rectangles):
    splitted = []
    for r in rectangles: 
        (x,y,w,h)=r 
        if w/h > 1.25: 
            new_w = int(w/2)
            splitted.append((x, y, new_w, h))
            splitted.append((x+new_w,y,new_w,h))
        else: 
            splitted.append(r)
    return splitted 

#extracts individual chars from the CAPTCHA image           
def getChar(letterBoundingRectangles, image): 
    extracted_char=[] 
    for r in letterBoundingRectangles:
        x,y,w,h=r
        single_char = image[y:y+h, x:x+w]
        extracted_char.append(single_char) 
    return extracted_char

#sorts by x coord. so each image will have the right label
def sortRectX(boundingRectangles): 
    return (sorted(boundingRectangles, key=lambda x: float(x[0])))

captcha_output_folder="/home/cas/Desktop/Licenta_App/extracted_chars"
char_dict={}
for captcha_single_img in captcha_dataset: 
    labels = get_label(captcha_single_img)
    captcha_img = read_img(captcha_single_img)
    grayscaled_img = grayscale(captcha_img)
    thresholded_img = threshold(grayscaled_img)
    dilated_img = dilate(thresholded_img)
    contours = findContours(dilated_img)
    splitted_chars = splitChars(calculateBoundingRectangles(contours))
    splitted_chars = sortRectX(splitted_chars)
    single_chars = getChar(splitted_chars, captcha_img)
    for single_char_img, current_chr in zip(single_chars,labels): 
        class_dir = os.path.join(captcha_output_folder, current_chr) 
        if not os.path.exists(class_dir): 
            os.makedirs(class_dir)
        char_count = char_dict.get(current_chr,1)
        single_img_path = os.path.join(class_dir, str(char_count) + ".png")
        cv2.imwrite(single_img_path, single_char_img) 
        char_dict[current_chr]=char_count+1
