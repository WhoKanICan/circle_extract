import numpy as np
from PIL import Image
import cv2
import pytesseract
import matplotlib.pyplot as plt
import os
import pdf2image
from collections import defaultdict
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


# TODO many more things I have yet to figure out how.
#  probably make use dictionary instead of lists

# TODO create robust function to work with linux and window file system for joining and splitting file_paths
def return_image(file_path: str):
    dir = os.path.dirname(os.path.realpath(file_path))
    try:
        os.mkdir(os.path.join(dir, 'processed images'))
    except FileExistsError:
        pass
    image_list = []
    new_dir = os.path.join(dir, 'processed images')
    images = pdf2image.convert_from_path(file_path)
    for idx, img in enumerate(images):
        img_path = os.path.join(new_dir, '{}_img.jpg'.format(idx))
        image_list.append(str(img_path))
        img.save(img_path, 'JPEG')

    return image_list, new_dir


def hsv_range(red: int, green: int, blue: int):
    '''
    packaged this persons converter.py script into a function. Yet to understand the maths
    https://henrydangprg.com/2016/06/26/color-detection-in-python-with-opencv/
    '''
    color = np.uint8([[[blue, green, red]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hue = hsv_color[0][0][0]
    hsv_lower = np.array([hue - 5, 100, 100], dtype=np.uint8)
    hsv_upper = np.array([hue + 5, 255, 255], dtype=np.uint8)
    return hsv_lower, hsv_upper


def center_from_contour(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    return center


# TODO clean up some of the var names in circle_function
def circle_location(image_list: list, hsv_range: list):
    """
    convert image to HSV and find contours for centre point location
    """
    mask_dict = dict()

    for idx, image in enumerate(image_list):
        dir = os.path.dirname(os.path.realpath(file_path))
        mask_path = os.path.join(dir, 'processed images', '{}_mask.jpg'.format(idx))

        img = cv2.imread(image, 1)  # read image as binary
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert BGR to HSV. openCV reads as BGR order not RGB
        mask = cv2.inRange(hsv_img, hsv_range[0], hsv_range[1])  # mask created by hsv range from hsv_range function
        cv2.imwrite(mask_path, mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # find contours of mask

        mask_dict[mask_path] = contours  # the mask path is the key and contour is the value

    return mask_dict


# TODO for some reason it detects shapes that are not masked as well! WHY!!!!
def pytesseract_OCR(mask_dict: dict, new_dir: str):
    segment_dict = dict()

    # preview_list = []
    # for i, (mask_image, contour_list) in enumerate(mask_dict.items()):
    #     masked_image = cv2.imread(mask_image)  # option argument 0 to read image as gray scale
    #     # img = cv2.drawContours(mask_image, contour_list)
    #     image_crop_path_list = []
    #     for idx, contour in enumerate(contour_list):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         # preview = cv2.rectangle(masked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         img_crop = masked_image[y:y + h, x:x + w]
    #         img_crop_path = os.path.join(new_dir, 'boundary_{}_{}.jpg'.format(i, idx))
    #         cv2.imwrite(img_crop_path, img_crop)
    #         preview_list.append(img_crop)
    #         image_crop_path_list.append(img_crop_path)
    #     segment_dict[mask_image] = image_crop_path_list
    #
    # config = r'--oem 2 --psm 4'
    # for mask_image, img_crop_path in segment_dict.items():
    #     for cropped_image in img_crop_path:
    #         img = cv2.imread(cropped_image)# option argument 0 to read image as gray scale
    #         image_height, image_width, _ = img.shape
    #         edges = cv2.Canny(img, 100, 200)
    #         img_new = Image.fromarray(edges)
    #
    #         boxes = pytesseract.image_to_data(img_new, lang='eng', config=config)
    #         print(boxes)
    #         for idx, box in enumerate(boxes.splitlines()):
    #             if idx != 0:
    #                 box = box.split()
    #                 if len(box) == 12:
    #                     x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
    #                     preview = cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 3)
    #                     preview_list.append(preview)

    preview_list = []
    config = r'--oem 3 --psm 4'
    for path, mask_img in mask_dict.items():
        img = cv2.imread(path)  # option argument 0 to read image as gray scale
        image_height, image_width, _ = img.shape
        edges = cv2.Canny(img, 100, 200)
        img_new = Image.fromarray(edges)

        boxes = pytesseract.image_to_data(img_new, lang='eng', config=config)

        for idx, box in enumerate(boxes.splitlines()):
            if idx != 0:
                box = box.split()
                if len(box) == 12:
                    print(box)
                    x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
                    preview = cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 3)
                    preview_list.append(preview)

    return preview_list


def resize(preview_list):
    for idx, preview in enumerate(preview_list):
        screen_res = 1280, 720
        scale_width = screen_res[0] / preview.shape[1]
        scale_height = screen_res[1] / preview.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(preview.shape[1] * scale)
        window_height = int(preview.shape[0] * scale)
        cv2.namedWindow('Resized Window_{}'.format(idx), cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Resized Window_{}'.format(idx), window_width, window_height)
        cv2.imshow('Resized Window_{}'.format(idx), preview)

        cv2.waitKey(0)


file_path = 'test_pdf.pdf'
images, path = return_image(file_path)
hsv_boundary = hsv_range(255, 255, 0)
mask_dict = circle_location(images, hsv_boundary)
text = pytesseract_OCR(mask_dict, new_dir=path)
resize(text)

# print(images)
# print(hsv_boundary)
# print(mask_path)
