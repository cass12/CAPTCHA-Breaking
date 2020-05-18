# Preprocessing the images using OpenCV2
# Extracting single character images from a 4 characters CAPTCHA image

import os
import cv2
import numpy as np

captcha_images_folder = "Desktop/captcha_images"
captchas = [
    os.path.join(captcha_images_folder, f) for f in os.listdir(captcha_images_folder)
]


def get_label(captcha_img):
    filename = os.path.basename(captcha_img)
    label = filename.split(".")[0]
    return label


def read_image(captcha_img):
    return cv2.imread(captcha_img)


def display(image)
    cv2.imshow("window", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def do_greyscale(captcha_image):
    return cv2.cvtColor(captcha_image, cv2.COLOR_BGR2GRAY)


def do_threshold(captcha_image_grayscaled):
    return cv2.threshold(
        captcha_image_grayscaled, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]


def dilate(binary_image):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(binary_image, kernel, iterations=1)


def find_contours(captcha_image_thresholded):
    return cv2.findContours(
        captcha_image_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]


def find_rectangles(contours):
    return list(map(cv2.boundingRect, contours))


def show_rectangles(rectangles, image):
    for rect in rectangles:
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    display(image)


def split_rectangles(rectangles):
    letter_bounding_rectangles = []
    for rectangle in rectangles:
        (x, y, w, h) = rectangle
        if w / h > 1.25:
            half_width = int(w / 2)
            letter_bounding_rectangles.append((x, y, half_width, h))
            letter_bounding_rectangles.append((x + half_width, y, half_width, h))
        else:
            letter_bounding_rectangles.append(rectangle)
    return letter_bounding_rectangles


def extract_char(rectangles, image):
    char_images = []
    for rect in rectangles:
        x, y, w, h = rect
        char_image = image[y - 1: y + h + 1, x - 1: x + w + 1]
        char_images.append(char_image)
    return char_images


def sort_rectangles(rects):
    return (sorted(rects, key=lambda x: float(x[0])))


captcha_processing_output_folder = "Desktop/extracted_characters"
character_counts = {}
for captcha_img in captchas:
    captcha_label = get_label(captcha_img)
    captcha_image = read_image(captcha_img)
    captcha_grayscaled = do_greyscale(captcha_image)
    captcha_thresholded = do_threshold(captcha_grayscaled)
    captcha_dilated = dilate(captcha_thresholded)
    captcha_contours = find_contours(captcha_dilated)
    character_bounding_rectangles = split_rectangles(find_rectangles(captcha_contours))
    character_bounding_rectangles = sort_rectangles(character_bounding_rectangles)
    character_images = extract_char(character_bounding_rectangles, captcha_image)
    for char_image, current_char in zip(character_images, captcha_label):
        if (len(character_images) == 4):
            save_dir = os.path.join(captcha_processing_output_folder, current_char)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            character_count = character_counts.get(current_char, 0)
            image_save_path = os.path.join(save_dir, str(character_count) + ".png")
            cv2.imwrite(image_save_path, char_image)
            character_counts[current_char] = character_count + 1
