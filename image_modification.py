import cv2 as cv
import os

__all__ = ["cut_face", "change_img_size", "rotate_img", "find_smallest_shape"]


def cut_face(img, coordinates: list):
    x, y, h, w = coordinates
    return img[y:y+h, x:x+w]

def change_img_size(img, shape: tuple):
    return cv.resize(img, shape)

def rotate_img(img, rotate_flag: str):
    """
        90: Counter ClockWise
        -90: ClockWise
        180: Symmetric
        
        for multiple rotation use + between flags
    """
    rotates_codes = {"90" : cv.ROTATE_90_COUNTERCLOCKWISE, "-90" : cv.ROTATE_90_CLOCKWISE, "180" :cv. ROTATE_180}
    
    if "+" in rotate_flag:
        commands = rotate_flag.split("+")
        for command in commands:
            img = cv.rotate(img, rotates_codes[command])
        return img
    
    return cv.rotate(img, rotates_codes[rotate_flag])

def find_smallest_shape(images):
    print(images[0].shape)
    smallest_col, smallest_row = images[0].shape[:2]

    for img in images[1:]:
        if img.shape[0] < smallest_row:
            smallest_row = img.shape[0]

        if img.shape[1] < smallest_col:
            smallest_col = img.shape[1]

    return (smallest_row, smallest_col)

