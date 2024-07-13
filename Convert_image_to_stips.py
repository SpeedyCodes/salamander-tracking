"""
Rune De Coninck

This document will try to convert an image of a salamander into a matrix where you have easy acces to the stips of
the salamander. We will do this in multiple steps.

1. Grayscale the image.
2. Threshold the image, with global, local mean and local Gaussian thresholds.
3. Blur the image and apply global thresholding to find some rough shapes, the middle shape will be from the salamander.
For this we use contour detection algorithm of OpenCV.
4. Get a new image where we only see this middle shape from the salamander. Thus, we removed background in the
blurred, global thresholding image. This gives a lot of challenges, we need to find the middle shape but also cut of
the edges if the middle shape, unfortunately, is connected with the edges. Also, the contour detection algorithm is not
always good, so in extreme cases we need to artificially cheat and create a universally new image.
5. [NOT DONE YET] Use this new image, where we only have the middle shape, to select a rough middle shape on a very
detailed image, like the images from adaptive mean and adaptive Gaussian thresholds. Thus, with this trick,
we actually have removed a lot of the annoying background in the good, detailed images, by using a very blurred and
global thresholding image.
6. [NOT DONE YET] Now we only have a detailed image of the salamander, with minimal background. We will now use a
method [TBC] to detect the stips in this image.
7. [NOT DONE YET] We will try to remove false positives and then add all the data in a matrix. This will be the end
of this file.
"""

import cv2 as cv
from Generating_extra_data import filenames_from_folder, resize_with_aspect_ratio
from matplotlib import pyplot as plt
import numpy as np

""" User input """
save_to_computer: bool = False  # Put this on True if you want to save the images while running this file.

path: str = 'C:/Users/Erwin2/OneDrive/Documenten/UA/Honours Program/Interdisciplinary Project/Salamanders/Edited images/'
# path refers to the place where you have all your different edited folders.
""" End user input """


def pre_processing(image, ksize):
    """ Pre-processing steps for image editing. """
    return cv.medianBlur(image, ksize)


def generate_thresholds(image):
    """ Generates global, local median and local Gaussian thresholds for an image. """

    # Documentation:
    # maxValue = non-zero value assigned to the pixels for which the condition is satisfied.
    # blockSize = size of a pixel neighborhood that is used to calculate a threshold value for the pixel: ...
    # 3, 5, 7, and so on.
    # C = constant subtracted from the mean or weighted mean. Normally, it is positive.

    image = pre_processing(image, 3)  # 127
    _, img_th_global = cv.threshold(image, 110, 255, cv.THRESH_BINARY)
    img_th_mean = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                       cv.THRESH_BINARY, 5, 4)
    img_th_Gaussian = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv.THRESH_BINARY, 5, 4)
    return img_th_global, img_th_mean, img_th_Gaussian


def plotting_images(image, img_th_global, img_th_mean, img_th_Gaussian):
    """ Plots a nice image of all different thresholds."""
    titles = ['Original Image', 'Global Thresholding',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    list_of_images = [image, img_th_global, img_th_mean, img_th_Gaussian]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(list_of_images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def draw_contours(image, img_th_global):
    """ Attempt to draw contours on an image, using the global threshold."""
    contours, _ = cv.findContours(img_th_global, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    image_copy = image.copy()
    index = -1
    thickness = 4
    color = (255, 0, 255)
    cv.drawContours(image_copy, contours, index, color, thickness)  # Draw the contours on the copied image.
    image_rescale = resize_with_aspect_ratio(image_copy, height=750)
    return image_rescale


def find_closest_white_point(image, center_x, center_y, window_size=30):
    """ Finds the closest white pixel to the center of the image, assuming that the input image is binaire.
    This is a helper function for isolate_central_object."""

    # Documentation:
    # Image needs to be binaire.
    # center_x and center_y are the coordinates of the center of the image.
    # Window_size denotes how far we may search for a white pixel, measured from the middle of the image.

    height, width = image.shape
    min_distance = float('inf')  # Minimum distance to the center of the image.
    closest_point = (center_x, center_y)

    # So, we only check the pixels around the center of the image, with a given window_size.
    for y in range(max(0, center_y - window_size), min(height, center_y + window_size)):
        for x in range(max(0, center_x - window_size), min(width, center_x + window_size)):
            if image[y, x] == 255:  # Checks if the point is white
                distance = (x - center_x) ** 2 + (y - center_y) ** 2  # Calculate the distance squared.
                if distance < min_distance:
                    min_distance = distance
                    closest_point = (x, y)
    return closest_point


def isolate_central_object(image, central_region_size=0.1, larger_square_size=0.4):
    """ Given a binaire image, who has a black background and white shapes. This method will return a new image,
    also with black background, but only with the most centered shape remaining.
    Thus, we remove all other white shapes."""

    # Documentation:
    # Image needs to be a binaire image with black background and white shapes.
    # Central_region_size denotes a percentage of the size of the image. We will need this to check if any contours run
    # through roughly the middle of the image, hence if they run through a central square around the center
    # of the image.
    # Larger_square_size is a percentage of the size of the image.
    # It is twofold, first it is to prevent fails. When the contours are badly drawn on the image,
    # due to bad quality of the image, we will sometimes get very strange results. Thus, we will just use a large white
    # square as middle shape instead. Second, if the middle shape is connected to other shapes near the edge of the
    # image, we want to remove these edges if they get outside the larger_square_size ratio.

    height, width = image.shape
    center_x, center_y = width // 2, height // 2

    # Check if the center point is white, otherwise find the closest white point.
    if image[center_y, center_x] != 255:
        center_x, center_y = find_closest_white_point(image, center_x, center_y)

    # Define the central region around the found central point.
    central_x1 = max(center_x - int(width * central_region_size / 2), 0)
    central_y1 = max(center_y - int(height * central_region_size / 2), 0)
    central_x2 = min(center_x + int(width * central_region_size / 2), width)
    central_y2 = min(center_y + int(height * central_region_size / 2), height)

    # Find the contours of the objects.
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create a mask for contours that run through the central region.
    mask = np.zeros_like(image)
    for contour in contours:
        if any(central_x1 < x < central_x2 and central_y1 < y < central_y2 for x, y in contour[:, 0, :]):
            cv.drawContours(mask, [contour], -1, [255], thickness=cv.FILLED)

    # Create a new image with only the central object isolated.
    isolated_image = np.zeros_like(image)
    isolated_image[mask == 255] = 255

    # Define the larger square region for making pixels outside black
    larger_x1 = max(center_x - int(width * larger_square_size / 2), 0)
    larger_y1 = max(center_y - int(height * larger_square_size / 2), 0)
    larger_x2 = min(center_x + int(width * larger_square_size / 2), width)
    larger_y2 = min(center_y + int(height * larger_square_size / 2), height)

    # Make pixels outside the larger square black, this corrects the cases when the middle shape is connected to
    # more shapes on the edges.
    isolated_image[:larger_y1, :] = 0  # Top
    isolated_image[larger_y2:, :] = 0  # Bottom
    isolated_image[:, :larger_x1] = 0  # Left
    isolated_image[:, larger_x2:] = 0  # Right

    # Check if isolated_image is almost fully black or almost fully white, if this is the case, an error has occurred
    # when computing the contours. In this case we make a universally new image,
    # this is a white square with black background.
    num_white_pixels = np.sum(isolated_image == 255)

    if num_white_pixels < 100 or num_white_pixels > (height * width - 100):  # Too much or too little white pixels.
        isolated_image = np.zeros_like(image)
        size = min(width, height)
        square_size = int(size * larger_square_size)

        x1 = (width - square_size) // 2
        y1 = (height - square_size) // 2
        x2 = x1 + square_size
        y2 = y1 + square_size
        isolated_image[y1:y2, x1:x2] = 255

    return isolated_image


if __name__ == '__main__':
    for edited_salamander in filenames_from_folder(f'{path}'):  # Looping over all edited folders.
        for number in filenames_from_folder(f'{path}/{edited_salamander}'):  # Looping over all edited salamanders.

            img = cv.imread(f'{path}/{edited_salamander}/{number}', 0)  # Grayscale is important!
            th_global, th_mean, th_Gaussian = generate_thresholds(img)

            # plotting_images(img, th_global, th_mean, th_Gaussian)

            # cv.imshow('Gaussian Thresholding', th_Gaussian)
            # cv.imshow('Global Thresholding', th_global)
            # cv.imshow('Median Thresholding', th_mean)

            # img_with_contours = draw_contours(img, th_global)
            # cv.imshow('Original with contours', img_with_contours)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            # img = pre_processing(img, 9)
            # _, img = cv.threshold(img, 110, 255, cv.THRESH_BINARY)
            # cv.imshow('blured version', img)
            # cv.waitKey(0)

            is_img = isolate_central_object(th_global)
            cv.imshow('Isolated', is_img)
            cv.imshow('Original', th_global)
            cv.waitKey(0)
            cv.destroyAllWindows()
            break

    # Still issues regarding .heic format:

    # for name in filenames_from_folder(f'{path}{location2024}'):
    #    show_image(name)
