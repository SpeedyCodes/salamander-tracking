"""
Rune De Coninck

This document will try to detect the dots on an image of a salamander.
Attention: this code is specifically made for the salamanders of year 2024!
"""

from isolate_salamander import isolate_salamander
import cv2 as cv
from Convert_image_to_stips import filenames_from_folder, resize_with_aspect_ratio
from utils.heic_imread_wrapper import wrapped_imread
import numpy as np
import copy


def convert_image_to_stips_v2024(image: np.ndarray, is_show: bool = False) -> set[tuple[int, int, int, int, bool]]:
    """ This function will include a bunch of other functions, it will run through the entire code

    INPUT: image in RGB format.
    INPUT: is_show = True will show the image.
    OUTPUT: list of coordinates of the dots, these are of the form:
    [ (x, y, w, h, is_good_dot), ... ]
    if is_good_dot is True then this is a good (green) dot, if it is false (red), then it is a false positive.
    The dots are being stored as rectangles where (x, y) are the coordinates of the top left corner of the dot.
    """

    image_full = copy.copy(image)
    image, image_gray, x_new, y_new = preprocess_image(image)
    list_of_dots = dot_detection(image, image_gray, x_new, y_new, min_area=20, max_area=2500)

    if is_show:
        show_image(image_full, list_of_dots)

    return list_of_dots


def preprocess_image(image: np.ndarray):
    """ This method preprocesses the given image. This includes isolating the salamander from the background.
    Thresholding and blurring and trying to preserve as many good edges as possible. """

    image = isolate_salamander(image)

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # x and y denote the linear transformation to the new origin after cropping.
    x, y, w, h = cv.boundingRect(image_gray)
    image = image[y: y + h, x: x + w]

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, image = cv.threshold(image_gray, 50, 255, cv.THRESH_BINARY)

    image = cv.medianBlur(image, 9)

    return image, image_gray, x, y


def dot_detection(image: np.ndarray, image_gray: np.ndarray, x_new: int, y_new: int, min_area, max_area):
    """ This method will detect the dots on an image of a salamander and label the dots as good dots of false dots.
    x_new and y_new denote the location of the new origin (after cropping) in regard to the original origin."""

    # Detect the largest contour in the image; this will be the body of the salamander.
    large_contours, _ = cv.findContours(image_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    salamander_contour = max(large_contours, key=cv.contourArea)

    # Detect the contours of the dots.
    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    height, width = image.shape[:2]
    min_dim = min(width, height)

    list_of_dots = set()

    def point_to_contour_distance(point):
        """Calculate the minimum distance from a point to the contour."""
        distances = cv.pointPolygonTest(salamander_contour, point, True)
        return abs(distances)

    for contour in contours:
        area = cv.contourArea(contour)
        if min_area < area < max_area:
            M = cv.moments(contour)

            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                x, y, w, h = cv.boundingRect(contour)

                # Check dot's area and distance to the salamander_contour.
                if area < 300 or w > 2 * h or h > 2 * w or w * h > 2500:  # Can be changed.
                    distance = point_to_contour_distance((cx, cy))
                    distance_threshold = min_dim * 0.1  # Can be changed.

                    if distance < distance_threshold:  # If dot is too small and close to the edge, then make it red.
                        color = (0, 0, 255)  # Red
                    else:
                        color = (0, 255, 0)  # Green
                else:
                    color = (0, 255, 0)  # Green

                if color == (0, 255, 0):
                    is_good_dot = True
                else:
                    is_good_dot = False
                list_of_dots.add((x + x_new, y + y_new, w, h, is_good_dot))

    return list_of_dots


def show_image(image: np.ndarray, list_of_dots):
    """ This method will show the given image with dots drawn on it. """

    for dot in list_of_dots:
        x, y, w, h, is_good_dot = dot

        if is_good_dot:
            image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow('Image with dots', resize_with_aspect_ratio(image))
    cv.waitKey(0)
    cv.destroyAllWindows()
    return None


year = '2024'

if __name__ == '__main__':
    for sal in filenames_from_folder(
            f'C:/Users/Erwin2/OneDrive/Documenten/UA/Honours Program/Interdisciplinary Project/Salamanders/{year}/'):  # Looping over all salamanders.
        img = wrapped_imread(
            f'C:/Users/Erwin2/OneDrive/Documenten/UA/Honours Program/Interdisciplinary Project/Salamanders/{year}/{sal}')

        cv.imshow('original', resize_with_aspect_ratio(img))
        cv.waitKey()

        list_dots = convert_image_to_stips_v2024(img, is_show=True)
