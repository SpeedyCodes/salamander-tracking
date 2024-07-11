"""
Rune De Coninck

This document will generate additional copies of existing images of our salamanders but with a change added.
Changes are made by rotations, scaling and adjusting brightness and contrast.
"""

import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import random

""" User input """
save_to_computer: bool = False  # Put this on True if you want to save the images while running this file.
amount_of_copies_per_salamander: int = 30  # Only useful if save_to_computer if True

path: str = 'C:/Users/Erwin2/OneDrive/Documenten/UA/Honours Program/Interdisciplinary Project/Salamanders/'
# Path refers to the place on your computer where you have two folders with names 2022 and 2024.

location2022: str = '2022/'
location2024: str = '2024/'
# These locations are the names of two folders. These folders respectively both contain the salamander images
# of 2022 and 2024.

# Thus for example, the location of a specific image should be like the following example:
# f'{path}{location2022}{2022-Sal02}'

""" End user input """


def filenames_from_folder(folder: str) -> set:
    """ Returns a list of all filenames in a folder."""
    list_of_images = set()
    for filename in os.listdir(folder):
        list_of_images.add(filename)
    return list_of_images


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv.INTER_AREA):
    """ Resizes the image with aspect ratio equal to width and height. """
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv.resize(image, dim, interpolation=inter)


def read_images(img_name: str):
    """ Reads images by their name."""
    if img_name[:4] == '2022':
        img = cv.imread(f'{path}{location2022}{img_name}', 0)  # Reads image from 2022.
    elif img_name[:3] == 'IMG':
        img = cv.imread(f'{path}{location2024}{img_name}', 0)  # Reads image from 2024.
    else:
        print('Image not found.')
        return None
    return img


def show_image(img_name: str):
    """ Shows the image. """
    img = read_images(img_name)
    img_resize = resize_with_aspect_ratio(img, height=750)
    cv.imshow(f'{img_name}', img_resize)
    cv.waitKey(0)

    cv.destroyAllWindows()


def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: float = 0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv.addWeighted(img, contrast, img, 0, brightness)


def change_image(img_name: str, angle: float, contrast: float, brightness: float):
    """ Rotates the image around the middle point. Furthermore, it scales the image."""
    # Reading in the image and changing contrast and brightness.
    img = read_images(img_name)
    img = adjust_contrast_brightness(img, contrast, brightness)

    # Constructing rotation matrix.
    h, w = img.shape[:2]
    img_center = (w / 2, h / 2)
    rot_mat = cv.getRotationMatrix2D(img_center, angle, 1)
    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    base_with = int((h * abs(sin)) + (w * abs(cos)))
    base_height = int((h * abs(cos)) + (w * abs(sin)))

    rot_mat[0, 2] += ((base_with / 2) - img_center[0])
    rot_mat[1, 2] += ((base_height / 2) - img_center[1])

    # Rotating the image.
    img_rotated = cv.warpAffine(img, rot_mat, (base_with, base_height), flags=cv.INTER_LINEAR)
    img_resize = resize_with_aspect_ratio(img_rotated, height=750)

    # Scaling the image.
    scale = random.uniform(0.6, 1.2)
    img_scaled = cv.resize(img_resize, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

    return img_scaled


if __name__ == '__main__':
    for name in filenames_from_folder(f'{path}{location2022}'):  # Looping over all images of a folder.
        for counter in range(1, amount_of_copies_per_salamander + 1):  # Making copies of a specific image.

            final_image = change_image(name, angle=360 * random.random(), contrast=random.uniform(0.5, 1.5),
                                       brightness=random.uniform(-30, 30))

            cv.imshow(f'{name}_changed_{counter}', final_image)  # Shows the image.

            if save_to_computer:
                name2 = name.removesuffix('.jpg')
                os.makedirs(f'{path}Edited Images/{name2}', exist_ok=True)  # Makes a folder to save the images.
                # It even works if the folder already existed.

                cv.imwrite(os.path.join(path, f'Edited Images/{name2}/{name}_changed_{counter}.jpg'), final_image)
                # Saves the images in the right folder (that was just made).

            cv.waitKey(0)

            cv.destroyAllWindows()

    # Still issues regarding .heic format:

    # for name in filenames_from_folder(f'{path}{location2024}'):
    #    show_image(name)
