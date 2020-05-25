# Preprocessing the images using OpenCV2
# Extracting single character images from a 4 characters CAPTCHA image

import os
import cv2
import numpy as np

#joing paths to get the image basename later on
#*it has a list format to be able to iterate trough all the labels
captcha_dataset_folder = "/home/cas/Desktop/Licenta_App/captcha_imgs" 
captcha_dataset_paths = [os.path.join(captcha_dataset_folder,x) for x in os.listdir(captcha_dataset_folder)] 


#extract the basename and remove extension to get the labels
def get_label(img_path):
    img_name = os.path.basename(img_path) 
    label = img_name.split(".")[0] 
    return label


#image to np array
def read_img(image): 
    image_to_array = cv2.imread(image)
    return image_to_array

def display(image): 
    cv2.imshow("CAPTCHA",image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows

def grayscale(image): 
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return gray

#returns retVal (Otsu optimal threshold) as first value
#so we take the second output which is the thresholded image 
def threshold(grayscaled_img): 
    thresholded = cv2.threshold(grayscaled_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    return thresholded

#convolves a kernel with the image in order to increase the white region 
def dilate(thresholded_img): 
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(thresholded_img, kernel, iterations = 1) 
    return dilated

#returns img, contours & hierarchy
#only the modified image is needed ([0])
def get_contours(thresholded_img): 
    contours = cv2.findContours(thresholded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours

#maps the function for each array(contour) to determine the contour for each letter
#ROI=region of interest
def get_ROI(contours): 
    boundingRectangles = map(cv2.boundingRect,contours)
    return list(boundingRectangles)

#rectangles are actually the bounding rectangles from countour
def display_rectangles(image,rectangles):
    for r in rectangles:
        x,y,w,h = r
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    display(image)
        

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

#extracts individual chars from the CAPTCHA image           
def extract_single_char(image, rectangles): 
    single_char = [] 
    for r in rectangles: 
        x,y,w,h = r
        single_char_img = image[y : y+h, x : x+w] 
        single_char.append(single_char_img) 
    return single_char

#sorts by x coord. so each image will have the right label (boundingRect can be get out of order) 
def sort_ROI(rectangles): 
    sorted_br = sorted(rectangles, key = lambda x: float(x[0]))
    return sorted_br

output_dataset_path = "/home/cas/Desktop/Licenta_App/extracted_chars1"
char_dict = {}
for captcha in captcha_dataset_paths:
    label = get_label(captcha)
    captcha_img = read_img(captcha)
    grayscaled_img = grayscale(captcha_img)
    thresholded_img = threshold(grayscaled_img)
    dilated_img = dilate(thresholded_img)
    contours = get_contours(dilated_img)
    bounding_rectangles = get_ROI(contours)
    splitted_chars = split_chars(bounding_rectangles)
    splitted_chars = sort_ROI(splitted_chars)
    single_chars = extract_single_char(captcha_img ,splitted_chars)
    for single_char_img, current_chr in zip(single_chars,label): 
        class_dir = os.path.join(output_dataset_path, current_chr) 
        if not os.path.exists(class_dir): 
            os.makedirs(class_dir)
        char_count = char_dict.get(current_chr,1)
        single_img_path = os.path.join(class_dir, str(char_count) + ".png")
        cv2.imwrite(single_img_path, single_char_img) 
        char_dict[current_chr]=char_count+1
