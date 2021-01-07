import numpy as np
from PIL import Image
import cv2
import pytesseract
import os
import pdf2image
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
        new_dir = os.mkdir(os.path.join(dir, 'processed images'))
    except FileExistsError:
        pass
    image_list = []
    new_dir = os.path.join(dir, 'processed images')
    images = pdf2image.convert_from_path(file_path)
    for idx, img in enumerate(images):
        img_path = os.path.join(new_dir, '{}_img.jpg'.format(idx))
        image_list.append(str(img_path))
        img.save(img_path, 'JPEG')

    return image_list


def hsv_range(red: int, green: int, blue: int):
    '''
    packaged this persons converter.py script into a function. Yet to understand the maths
    https://henrydangprg.com/2016/06/26/color-detection-in-python-with-opencv/
    '''
    color = np.uint8([[[blue, green, red]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hue = hsv_color[0][0][0]
    hsv_lower = np.array([hue - 10, 100, 100], dtype=np.uint8)
    hsv_upper = np.array([hue + 10, 255, 255], dtype=np.uint8)
    return hsv_lower, hsv_upper


# TODO clean up some of the var names in circle_function
def circle_location(image_list: list, hsv_range: list):
    """
    convert image to HSV and find contours for centre point location
    """

    location_list, contour_list, mask_list = [], [], []
    for idx, image in enumerate(image_list):
        dir = os.path.dirname(os.path.realpath(file_path))
        mask_path = os.path.join(dir, 'processed images', '{}_mask.jpg'.format(idx))
        mask_list.append(mask_path)

        img = cv2.imread(image, 1)  # read image as binary
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert BGR to HSV. opencv reads as BGR order not RGB
        mask = cv2.inRange(hsv, hsv_range[0], hsv_range[1])  # mask created by hsv range from hsv_range function
        cv2.imwrite(mask_path, mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)# find contours of mask

        location_sublist, contour_sublist = [], []
        for circle in contours:  # get contour of each circle of mask
            (x, y), radius = cv2.minEnclosingCircle(circle)
            center = (int(x), int(y))
            location_sublist.append(center)  # add x,y points to location list

        location_list.append(location_sublist)
        contour_list.append(contours)

    return location_list, mask_list, contour_list

# TODO for some reason it detects shapes that are not masked as well! WHY!!!!
def pytesseract_OCR(mask_list: list, contour_list: list):

    preview_list = []
    for masked_image in mask_list:
        masked_image = cv2.imread(masked_image)  # option argument 0 to read image as gray scale
        for idx, contour in enumerate(contour_list):
            for idxx, contours in enumerate(contour):
                x, y, w, h = cv2.boundingRect(contour[idxx])
                preview = cv2.rectangle(masked_image,(x,y),(x+w,y+h),(0,255,0),2)
                img_crop = masked_image[y:y + h, x:x + w]
                cv2.imwrite('processed images/boundary_{}_{}.jpg'.format(idx,idxx), img_crop)
                preview_list.append(preview)

    return preview_list
    #
    # # OCR recognition
    # preview_list, box_list = [], []
    # for masked_image in mask_list:
    #     img = cv2.imread(masked_image)  # option argument 0 to read image as gray scale
    #     image_height, image_width, _ = img.shape
    #     edges = cv2.Canny(img, 100, 200)
    #     img_new = Image.fromarray(edges)
    #     boxes = pytesseract.image_to_data(img_new, lang='eng')
    #
    #     for idx, box in enumerate(boxes.splitlines()):
    #         if idx != 0:
    #             box = box.split()
    #             if len(box) == 12:
    #                 box_list.append(box)
    #                 x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
    #                 preview = cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 3)
    #                 preview_list.append(preview)

    #
    # return box_list, preview_list



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
images = return_image(file_path)
hsv_boundary = hsv_range(255, 255, 0)
circle_centres, mask_path, contour_list = circle_location(images, hsv_boundary)
text = pytesseract_OCR(mask_path, contour_list)
resize(text)


# resize(preview_list)
#
# print(images)
# print(hsv_boundary)
# print(mask_path)


