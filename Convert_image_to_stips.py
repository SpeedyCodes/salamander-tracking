"""
Rune De Coninck

This document will try to convert an image of a salamander into a matrix where you have easy acces to the stips of
the salamander. We will do this in multiple steps.

1. Grayscale the image.
2. Threshold the image, with global, local mean and local Gaussian thresholds.
3. Blur the image and apply global thresholding to find some rough shapes, the middle shape will be from the salamander.
For this we use contour detection algorithm of OpenCV.
4.
OLD VERSION:
Get a new image where we only see this middle shape from the salamander. Thus, we removed background in the
blurred, global thresholding image. This gives a lot of challenges, we need to find the middle shape but also cut of
the edges if the middle shape, unfortunately, is connected with the edges. Also, the contour detection algorithm is not
always good, so in extreme cases we need to artificially cheat and create a universally new image.
NEW VERSION:
We use the code from isolate_salamander.py to find and isolate this middle shape.
5. Use this new image, where we only have the middle shape, to select a rough middle shape on a very
detailed image, like the images from adaptive mean and adaptive Gaussian thresholds. Thus, with this trick,
we actually have removed a lot of the annoying background in the good, detailed images, by using a very blurred and
global thresholding image.
6. Now we only have a detailed image of the salamander, with minimal background. We will now use a
method to detect the stips in this image.
7. We will try to remove false positives and then add all the data in a matrix. This will be the end
of this file.
8. [NOT DONE YET] Optimize the program and improve all the steps such that the program can handle images with worse
quality and can easily detect false positive dots.
"""

import cv2 as cv
from Generating_extra_data import filenames_from_folder, resize_with_aspect_ratio
from isolate_salamander import isolate_salamander
from matplotlib import pyplot as plt
import numpy as np
from utils.heic_imread_wrapper import wrapped_imread

""" User input """
save_to_computer: bool = False  # Put this on True if you want to save the images while running this file.

path: str = ('C:/Users/Erwin2/OneDrive/Documenten/UA/Honours Program/Interdisciplinary Project/Salamanders/'
             'Edited images/2022/')
# path refers to the place where you have all your different edited folders.

old_new: bool = True  # True for new version, False for old version.
""" End user input """


def convert_image_to_coordinate_stips(image: np.ndarray, version: bool = True) -> set[tuple[int, int, int, bool]]:
    """ This function will include a bunch of other functions, it will return the coordinates of the good stips
    on the belly of the salamander.

    INPUT: numpy array image (this can be in BGR format), this type of image can be obtained by wrapped_imread.
    INPUT: boolean, False will give old version, True will give new version.
    OUTPUT: set with 4-tupels: (x-coordinate of center dot, y-coordinate of center dot, radius of dot,
    good or bad dot: this is a boolean)."""

    if not version:  # Old version

        # Conversion to grayscale, this is important!
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Generating local mean threshold.
        _, image_th_mean, _ = generate_thresholds(image)

        # Pre-processing of the image. ksize = 7 is found experimentally.
        image_blur = pre_processing(image, 7, None, True)
        _, image_blur_th_global = cv.threshold(image_blur, 110, 255, cv.THRESH_BINARY)

        # Isolating central object.
        image_isolate = isolate_central_object(image_blur_th_global)

    else:  # New version

        image_isolate = isolate_salamander(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, image_th_mean, _ = generate_thresholds(image)

    # Cropping the image.
    image_crop_original_size = crop_detailed_image_original_size(image_th_mean, image_isolate)

    # Post-processing of the image.
    image_post_proc = post_processing(image_crop_original_size, 5, 180)

    # Detecting the dots.
    list_of_dots = detect_dots(image_post_proc)

    return list_of_dots


def pre_processing(image, ksize, stddev, full_blur: bool):
    """ Pre-processing steps for image editing. These are the steps done in the beginning of the proces. """

    # Full_blur is True will use the standard, rough, blur (is more useful for finding a rough shape of the salamander).
    # Full_blur is False will use more advanced methods, like Gaussian blur.

    if full_blur:
        image_divided = cv.blur(image, (ksize, ksize))
        return image_divided

    image_blurred = cv.GaussianBlur(image, (ksize, ksize), sigmaX=stddev, sigmaY=stddev)
    image_divided = cv.divide(image, image_blurred, scale=255)  # Dividing gives a better background with less noise.
    return image_divided


def post_processing(image, kernel_size, threshold):
    """ Post-processing steps for image editing. These are the steps done at the end of the proces.
    When we have a detailed image of the belly of the salamander, we need to group neighbouring small stips into
    big stips and remove noise."""

    # Pure blurring the image such that we get nice big dots.
    image_blur = cv.blur(image, (kernel_size, kernel_size))

    # Remove noise.
    _, image_blur_threshold = cv.threshold(image_blur, threshold, 255, cv.THRESH_BINARY)

    return image_blur_threshold


def generate_thresholds(image):
    """ Generates global, local median and local Gaussian thresholds for an image. """

    # Documentation:
    # maxValue = non-zero value assigned to the pixels for which the condition is satisfied.
    # blockSize = size of a pixel neighborhood that is used to calculate a threshold value for the pixel: ...
    # 3, 5, 7, and so on.
    # C = constant subtracted from the mean or weighted mean. Normally, it is positive.

    image_copy = image.copy()
    image_copy = pre_processing(image_copy, 77, 77, False)

    image = cv.medianBlur(image, 3)
    _, img_th_global = cv.threshold(image, 110, 255, cv.THRESH_BINARY)  # Use less pre_processing.

    img_th_mean = cv.adaptiveThreshold(image_copy, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                       cv.THRESH_BINARY, 5, 4)
    img_th_Gaussian = cv.adaptiveThreshold(image_copy, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
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


def isolate_central_object(image, central_region_size=0.15, larger_square_size=0.4):
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
    """ MAYBE CHANGE "RETR_STYLE" LATER ON IF THAT SEEMS BETTER!"""
    contours, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # Create a mask for contours that run through the central region.
    mask = np.zeros_like(image)

    for contour in contours:
        if any(central_x1 < x < central_x2 and central_y1 < y < central_y2 for x, y in contour[:, 0, :]):
            cv.drawContours(mask, [contour], -1, [255], thickness=cv.FILLED)

    # Create a new image with only the central object isolated.
    isolated_image = np.zeros_like(image)
    isolated_image[mask == 255] = 255

    cv.imshow('isolated image', isolated_image)
    cv.waitKey()

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

    # In the case that there are too much or too little white pixels.
    limit = 1200
    ellipse_check = True  # Checks if it needs to draw an ellips or not. In the latter case, it will draw a rectangle.
    if num_white_pixels < limit or num_white_pixels > (height * width - limit):
        ellipse_check = False
        isolated_image = np.zeros_like(image)
        size = min(width, height)
        square_size = int(size * larger_square_size)

        x1 = (width - square_size) // 2
        y1 = (height - square_size) // 2
        x2 = x1 + square_size
        y2 = y1 + square_size
        isolated_image[y1:y2, x1:x2] = 255

    # Now we will try to draw a nice figure, like an ellips or rectangle, around the white middle shape.
    contours_of_middle_shape, _ = cv.findContours(isolated_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    surrounded_image = np.zeros_like(image)

    # Combine all existing contours on isolated_image.
    all_contours_combined = np.concatenate(contours_of_middle_shape)

    if ellipse_check:
        try:
            ellipse = cv.fitEllipse(all_contours_combined)
            # Increase the size of the ellipse by scaling its axes with 10%, this ensures that we don't forget anything.
            ellipse = (ellipse[0], (ellipse[1][0] * 1.1, ellipse[1][1] * 1.1), ellipse[2])
            cv.ellipse(surrounded_image, ellipse, (255, 255, 255), 2)
        except cv.error:  # Error? Then we will draw rectangle.
            x, y, w, h = cv.boundingRect(all_contours_combined)
            cv.rectangle(surrounded_image, (x, y), (x + w, y + h), (255, 255, 255), 2)

    if not ellipse_check:  # Draw a rectangle around the middle shape.
        x, y, w, h = cv.boundingRect(all_contours_combined)
        cv.rectangle(surrounded_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return surrounded_image


def crop_detailed_image_original_size(detailed_image, surrounded_image):
    """ This method will change the detailed image using the shape of the surrounded image of the method
     isolate_central_object, since both have the same dimensions. We will overlap the detailed image and the
     surrounded image and only leave the part of the detailed image unchanged that is inside the shape.
     All the pixels on the outside of the image, we change to white. Thus, we return an image with a white background
     and a detailed fraction of the detailed image in the center."""

    contours, _ = cv.findContours(surrounded_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create a mask, this is a full white image.
    mask = np.ones_like(surrounded_image) * 255

    cv.drawContours(mask, contours, -1, color=[0], thickness=cv.FILLED)  # Make the inside of the shape black,
    # We do this to ensure that this part doesn't get cropped.

    cropped_image_original_size = np.where(mask == 255, 255, detailed_image)  # Make the outside of the shape white
    # on the detailed_image.

    return cropped_image_original_size


def crop_detailed_image_small_size(cropped_image_original_size):
    """ This method will literally crop the cropped image. So we remove the non-interesting, white background that we
    have created in the method crop_detailed_image_original_size. We return an image with smaller size than the original
    image, but this smaller image only contains key information of the salamander, no extra white background."""

    # We will first need to find the bounding box of the middle shape.
    # To be able to do this, we first need to change white to black and black to white.
    cropped_image_original_size = cv.bitwise_not(cropped_image_original_size)

    x, y, w, h = cv.boundingRect(cropped_image_original_size)

    # Now crop the image to the bounding box.
    cropped_image = cropped_image_original_size[y:y + h, x:x + w]

    # Again reverse the colors back to what we are used to.
    cropped_image = cv.bitwise_not(cropped_image)

    return cropped_image


def gaussian_weight(distance, sigma):
    """ Returns the value of the Gaussian weight function. This is needed for better results in detect_dots."""

    # Documentation:
    # Distance is the distance from the center of the dot to the middle of the image.
    # Sigma controls the width of the Gaussian curve.

    weight = np.exp(-(distance ** 2) / (2 * sigma ** 2))

    return weight


def detect_dots(image, min_area=10, max_area=400, sigma_divider=6):
    """ This function will detect the dots on the cropped image of the belly of the salamander.
    Next it will try to isolate the good dots and bad dots from each other.
    Good dots are dots that are indeed dots of the salamander, these have a lot of white around them.
    Bad dots are false positives and come from the noise in the background. Mostly these have a lot of black around
    them, so we can isolate the dots on this fact: the percentage of white and black around the dots.
    To be more accurate, we will change the threshold (this percentage), for what we call a good or bad dot based
    on the location on the image. If the dot is near the edge of the image, it is more likely from noise and hence we
    can lower our threshold, thus we need less other black noise around it to consider it a bad dot. While dots in the
    center are more likely to be good dots, thus these need more black around them to be labeled false positives and
    hence bad dots. This computation of the threshold is done such that the threshold is continuous. This will give
    the best results and that's why we use the Gaussian weight function.

    This function returns a set which consists of 4-tupels with cartesian coordinates of the center point and the
    radius and if the dot is good or bad."""

    # Documentation:
    # min_area is the minimum area of what a dot can be.
    # max_area is the maximum area of what a dot can be.
    # Sigma_divider is used for the Gaussian_weight function. It helps in the following manner:
    # If too many dots far from the center are classified as good, then increase the sigma_divider.
    # If too many dots near the center are classified as bad, then decrease the sigma_divider.

    # Output will be an image with green (good) dots and red (bad) dots.

    # Find contours, we use RETR_TREE for more accurate results. MAYBE CHANGE LATER.
    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    list_of_dots = set()

    # Processing of each dot, we calculate the center of the dots, using cv.moments.
    for contour in contours:
        area = cv.contourArea(contour)
        if min_area < area < max_area:
            M = cv.moments(contour)

            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                radius = int(np.sqrt(area / np.pi))

                distance_from_center_image = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

                sigma = width / sigma_divider
                weight = gaussian_weight(distance_from_center_image, sigma)

                # Calculate continuous threshold percentage based on weight. MAY BE CHANGED IN THE FUTURE!
                min_threshold_percentage = 15
                max_threshold_percentage = 45
                threshold_percentage = min_threshold_percentage + weight * (
                        max_threshold_percentage - min_threshold_percentage)

                # Try to compute how much black is around the dots.
                # Do this by making a new bigger circle, around the existing one, thus we also will make two masks.
                original_circle_mask = np.zeros_like(image)
                larger_circle_mask = np.zeros_like(image)
                cv.circle(original_circle_mask, (cx, cy), radius, [255], -1)
                cv.circle(larger_circle_mask, (cx, cy), 2 * radius, [255], -1)

                # Create intermediate mask which is the area between larger circle and original circle.
                intermediate_mask = cv.bitwise_xor(larger_circle_mask, original_circle_mask)

                # Calculate percentage of white and black pixels within the larger circle
                total_pixels = np.count_nonzero(intermediate_mask)
                white_pixels = np.count_nonzero(intermediate_mask & image)
                black_pixels = total_pixels - white_pixels
                black_percentage = (black_pixels / total_pixels) * 100

                list_of_dots.add((cx, cy, radius, black_percentage < threshold_percentage))

    return list_of_dots


def draw_dots(image, list_of_dots):
    """ This method will draw green dots (good dots) and red dots (false positives) on an image. """

    image_with_dots = cv.cvtColor(image, cv.COLOR_GRAY2BGR)  # We want color for green and red dots.
    for dot in list_of_dots:
        cx, cy, radius, is_good_dot = dot

        # Color the dot based on black percentage; the threshold
        if is_good_dot:
            cv.circle(image_with_dots, (cx, cy), radius, (0, 255, 0), -1)  # Green
        else:
            cv.circle(image_with_dots, (cx, cy), radius, (0, 0, 255), -1)  # Red

    return image_with_dots


def image_to_matrix(image, list_of_dots):
    """ This method will use a list of coordinates of dots from a salamander image. Then it will convert these
      coordinates to entries of a matrix. These entries will have value 1 in the matrix if the pixel corresponding
      to that entry lies within a good dot. The other entries have value 0."""

    height, width = image.shape[:2]
    matrix = np.zeros((height, width), dtype=int)

    for cx, cy, radius, is_good in list_of_dots:
        if is_good:  # Only care about the good dots.
            for x in range(cx - radius, cx + radius + 1):
                for y in range(cy - radius, cy + radius + 1):
                    # Check if it is indeed a pixel within the circle.
                    if 0 <= x < width and 0 <= y < height and (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                        matrix[y, x] = 1  # Because of strange indexing.

    return matrix


def display_matrix(matrix):
    """ Displays the matrix in a nice way."""

    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap='Greys', interpolation='nearest')
    plt.title('Matrix Representation of Green Dots')
    plt.colorbar()
    plt.show()


year = '2024'

if __name__ == '__main__':
    for sal in filenames_from_folder(
            f'C:/Users/Erwin2/OneDrive/Documenten/UA/Honours Program/Interdisciplinary Project/Salamanders/{year}/'):  # Looping over all salamanders.
        img = wrapped_imread(
            f'C:/Users/Erwin2/OneDrive/Documenten/UA/Honours Program/Interdisciplinary Project/Salamanders/{year}/{sal}')

        img_isolate_new = isolate_salamander(img)
        cv.imshow('New version', resize_with_aspect_ratio(img_isolate_new, height=750))
        cv.waitKey()
        cv.destroyAllWindows()

        """ Loading in the image."""
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Grayscale is important!

        """ Generating different thresholds. """
        th_global, th_mean, th_Gaussian = generate_thresholds(img)
        # plotting_images(img, th_global, th_mean, th_Gaussian)
        # cv.imshow('Gaussian Thresholding', th_Gaussian)
        # cv.imshow('Global Thresholding', th_global)
        # cv.imshow('Median Thresholding', th_mean)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        if not old_new:  # Old version

            """ Pre-processing of the image. ksize = 7 is found experimentally."""
            img_blur = pre_processing(img, 7, None, True)
            cv.imshow('Pre-processing', img_blur)
            _, img_blur_global = cv.threshold(img_blur, 110, 255, cv.THRESH_BINARY)
            cv.imshow('Binaire', img_blur_global)

            """ Isolating central object."""
            isolate_img = isolate_central_object(img_blur_global)
            cv.imshow('Isolated', isolate_img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        else:  # New version
            isolate_img = cv.cvtColor(img_isolate_new, cv.COLOR_BGR2GRAY)

        """ Cropping the image."""
        img_crop_original_size = crop_detailed_image_original_size(th_mean, isolate_img)
        cv.imshow('Cropped_original_size', img_crop_original_size)
        img_crop = crop_detailed_image_small_size(img_crop_original_size)
        cv.waitKey()
        cv.imshow('Cropped', img_crop)

        """ Post-processing of the image."""
        img_post_proc = post_processing(img_crop, 5, 180)
        cv.imshow('Image post-processing', img_post_proc)

        """ Detecting the dots."""
        dots = detect_dots(img_post_proc)
        img_dots = draw_dots(img_crop, dots)
        cv.imshow('Dots', img_dots)

        """ Representing the good dots in a matrix and showing this matrix."""
        matrix_with_dots = image_to_matrix(img_dots, dots)
        display_matrix(matrix_with_dots)

        cv.waitKey(0)
        cv.destroyAllWindows()
        