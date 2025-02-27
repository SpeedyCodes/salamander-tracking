""" Rune De Coninck

Given a picture, this document will try to isolate (the belly of) the salamander from the background.
Hence, it will return a new image, without minimal background.
There are two main ways of dealing with this, first the old way and second the new way.
By using two approaches, we can have a buffer for when a specific method does not work on a given image.

1. By using color segmentation and morphological operations to detect the salamander.
After that, we will try to remove some parts of the background.

2. Using pose estimation to detect points on the torso of the salamander. Then we interpolate those points and
remove everything outside the perimeter.
"""

import cv2 as cv
import numpy as np
import largestinteriorrectangle as lir
from typing import Dict, Tuple
import math
from scipy.interpolate import splprep, splev
from config import pose_estimation_confidence
import warnings


def assert_bgr_format(image):
    assert isinstance(image, np.ndarray), "Image must be a numpy array."
    assert image.ndim == 3, "Image must be in BGR format."
    assert image.shape[2] == 3, "Image must be in BGR format."


def assert_pose_estimation_evaluation(val):
    assert 0 <= val <= 2, "Pose estimation evaluation must be between 0 and 2."


"""
Pipeline isolate_salamander.py.
"""


def isolate_salamander(image: np.array, coordinates_pose: Dict[str, Tuple[int, int, float]] | None = None,
                       is_background_removed: bool = False, pose_estimation_evaluation: int = 0) -> np.array:
    """ This function will include a bunch of other functions.

    INPUT: numpy array image, this must! be in BGR format, this type of image can be obtained by wrapped_imread.
    Important is that this image can already have removed background!
    INPUT: coordinates_pose are the coordinates of the important parts of the body of the salamander, detected
    by the pose estimation.
    INPUT: is_background_removed denotes if the background of the image is already removed or not.
    INPUT: pose_estimation_evaluation, this will be 0, 1 or 2 depending on how good the pose estimation worked.
        Value = 0 means the pose estimation failed twice, or we do not want to use it. So we use old methods.
        Value = 1 means the pose estimation failed once, and we will now remove the background and try again.
        Value = 2 means the pose estimation succeeded from the first time, and we will use it to find the belly.

    OUTPUT: A new image that has a black background. The only thing remaining is (the belly of) the salamander. """

    value = pose_estimation_evaluation

    assert_bgr_format(image)
    assert_pose_estimation_evaluation(value)

    if value == 0:  # We use old methods to find the belly.

        if is_background_removed:
            image_belly = find_belly_with_old_methods(image, is_background_removed=True)

        else:
            image_belly = find_belly_with_old_methods(image, is_background_removed=False)

        return image_belly

    elif value == 1:  # We need to remove the background, but leave salamander intact.
        image_no_background, _ = remove_background_old_methods(image)
        return image_no_background

    elif value == 2:  # We need to isolate the belly of the salamander using pose estimation.
        tck, points = find_torso(coordinates_pose)

        image_belly_with_pose_estimation = remove_everything_outside_curve(image, tck)

        return image_belly_with_pose_estimation


"""
1. Old method: Color Segmentation.

(a) Removing the background.
"""


def remove_background_old_methods(image: np.array) -> np.array:
    """ This function will remove the background on the image using Color Segmentation.
    It will return an image with only the salamander remaining in a black background. """

    # First, detect the salamander (and mask) with color segmentation, thus there will be some noise left.
    image_isolated_salamander_with_noise, mask = color_segmentation(image, ksize=51, lower_bound=[5, 50, 50],
                                                                    upper_bound=[35, 255, 255])

    # Try to filter out the noise based on the fact that the contour of the salamander is a big central object.
    image_contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    height, width = image.shape[:2]
    central_x = width // 2
    central_y = height // 2

    # Filter the detected contours and find the one of the salamander.
    filtered_contours, contour_with_largest_area = filter_contours(image_contours, width, height,
                                                                   parameter_for_central_radius=4,
                                                                   central_x=central_x, central_y=central_y)
    best_contour = select_best_contour(filtered_contours, central_x, central_y, contour_with_largest_area)

    # Isolate the whole salamander from the background.
    image_isolated_without_noise = draw_best_contour(best_contour, image_isolated_salamander_with_noise)

    return image_isolated_without_noise, best_contour


def find_belly_with_old_methods(image: np.array, is_background_removed: bool) -> np.array:
    """ This function find the belly with old methods, this function is only used when we already used
    Color Segmentation."""

    best_contour = None
    if not is_background_removed:  # Then first remove it...
        image, best_contour = remove_background_old_methods(image)

    # Try to find the belly of the salamander.
    image_belly = find_the_belly(best_contour, image)

    return image_belly


def color_segmentation(image, ksize, lower_bound, upper_bound):
    """ Detects everything on the image that has the same color as the salamander, including the salamander."""

    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)  # Convert to HSV color space for more practical usage.

    # Define colors of the salamander.
    lower_color = np.array(lower_bound)  # Lower bound in HSV values.
    upper_color = np.array(upper_bound)  # Upper bound in HSV values.

    mask_color = cv.inRange(hsv_image, lower_color, upper_color)

    # Clean the mask:
    mask_color = clean_mask(mask_color, ksize)

    image_with_mask = cv.bitwise_and(image, image, mask=mask_color)  # Apply mask to image.
    return image_with_mask, mask_color


def clean_mask(mask, ksize):
    """ This method cleans the mask using morphological operations."""

    kernel = np.ones((5, 5), np.uint8)

    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)  # Closing some holes in the mask.
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)  # Removing some small noise.

    # Smooth the edges of the mask, the higher ksize, the smoother the edges.
    mask = cv.GaussianBlur(mask, (ksize, ksize), 0)

    _, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)  # Making it binary again.

    # If the salamander is connected to some noise, the following steps will separate the salamander and the noise.
    kernel = np.ones((30, 30), np.uint8)
    mask = cv.erode(mask, kernel, iterations=2)  # Separate connected components.
    mask = cv.dilate(mask, kernel, iterations=2)  # Restore the shape of the salamander.

    return mask


def central_radius(parameter, width, height):
    """ This function will return the length of a radius around the center of the image."""

    return int(min(width, height) / parameter)


def is_contour_close_to_edge(contour, width, height, edge_percentage=0.1, required_percentage=0.15):
    """ Returns True if the contour is close to edge, False otherwise.
    edge_percentage is what we call the edge, and it is the percentage with respect to the size of the image.
    If more than required_percentage is close to edge, we will consider it close to the edge."""

    edge_x = width * edge_percentage
    edge_y = height * edge_percentage

    edge_point_count = 0  # Number of points that are within the edge area.

    for point in contour:
        x, y = point[0]

        if (x < edge_x or x > width - edge_x or
                y < edge_y or y > height - edge_y):
            edge_point_count += 1  # Since the point is in the edge.

    edge_point_proportion = edge_point_count / len(contour)

    return edge_point_proportion >= required_percentage


def intersects_central_square(contour, radius, central_x, central_y):
    """ This method checks if a contour intersects with the central part of the image.
    It returns True if the contour intersects with the central part of the image, False otherwise."""

    x, y, w, h = cv.boundingRect(contour)
    contour_rect = (x, y, x + w, y + h)
    square_rect = (central_x - radius, central_y - radius,
                   central_x + radius, central_y + radius)

    # Check if the bounding rectangle intersects with the central square.
    return (not (contour_rect[2] < square_rect[0] or
                 contour_rect[0] > square_rect[2] or
                 contour_rect[3] < square_rect[1] or
                 contour_rect[1] > square_rect[3]))


def contour_perimeter(contour):
    """ This method returns the perimeter of a contour."""

    return cv.arcLength(contour, True)


def contour_area(contour):
    """ This method returns the area of a contour."""

    return cv.contourArea(contour)


def closest_point_to_center(contour, central_x, central_y):
    """ This method finds the closest point to the center of the image, on a given contour."""

    min_distance = float('inf')

    contour = [points for sub_contour in contour for points in sub_contour]  # Flattens the list.
    for point in contour:
        x, y = point[0]
        distance = np.sqrt((x - central_x) ** 2 + (y - central_y) ** 2)

        if distance < min_distance:
            min_distance = distance

    return min_distance


def filter_contours(contours, width, height, parameter_for_central_radius, central_x, central_y):
    """ This method will filter out the bad contours and hopefully only return the contour of the salamander."""

    assert len(contours) > 0, 'No contours found!'
    contours = filter_contours_placement(contours, width, height, parameter_for_central_radius, central_x, central_y)
    contours = filter_contours_concentric(contours)
    contours, contour_with_largest_area = filter_contours_area(contours, area_ratio_threshold=0.30)

    return contours, contour_with_largest_area


def filter_contours_placement(contours, width, height, parameter_for_central_radius, central_x, central_y):
    """ This method will first filter the contours on their placement. We only keep the contours in the center of the
    image and not near the edge."""

    radius = central_radius(parameter_for_central_radius, width, height)
    good_contours = []

    for contour in contours:
        if intersects_central_square(contour, radius, central_x, central_y) and (not is_contour_close_to_edge(contour,
                                                                                                              width,
                                                                                                              height)):
            good_contours.append(contour)

    # It could be that our restrictions are too strict, then we need to increase the radius.
    if len(good_contours) == 0:
        p = parameter_for_central_radius

        assert p > 3

        radius = central_radius(1 / (p - 1), width, height)
        for contour in contours:
            if intersects_central_square(contour, radius, central_x, central_y) and (
                    not is_contour_close_to_edge(contour, width, height)):
                good_contours.append(contour)

        if len(good_contours) == 0:
            radius = central_radius(1 / (p - 2), width, height)
            for contour in contours:
                if intersects_central_square(contour, radius, central_x, central_y) and (
                        not is_contour_close_to_edge(contour, width, height)):
                    good_contours.append(contour)

        # If no contours are in the middle of the image, then we need to consider all contours.
        if len(good_contours) == 0:
            good_contours = [c for c in contours]

    return good_contours


def filter_contours_concentric(good_contours):
    """ Often, the salamander is put in a box, and we detect the contour of the salamander and the contour of the box.
    In this case we will have two concentric shapes. Therefore, we will remove the biggest concentric shape,
    if it exists. """

    sorted_contours = sorted(good_contours, key=contour_perimeter, reverse=True)  # Finding the biggest one.
    biggest = sorted_contours[0]

    # Drawing an ellipse around the biggest contour. Now we just need to check if the other contours completely lie
    # in this ellipse. If this is True, then we remove the biggest contour.
    assert len(biggest) != 4, 'Only contour found is the trivial one, please try again with a better image!'
    assert len(biggest) >= 5, 'No good contours were found!'
    ellipse = cv.fitEllipse(biggest)
    (center, axes, angle) = ellipse
    (major_axis, minor_axis) = axes
    (cx, cy) = center

    if len(sorted_contours) > 1:
        is_concentric = True
    else:
        is_concentric = False

    for contour in sorted_contours[1:]:  # Do not consider the biggest contour because this would be useless.

        # Check if each point in the contour is within the ellipse or not.
        for point in contour:
            px, py = point[0]

            # First do some transformations in Euclidean space.
            cos_angle = np.cos(np.radians(-angle))
            sin_angle = np.sin(np.radians(-angle))
            dx = px - cx
            dy = py - cy
            tx = dx * cos_angle - dy * sin_angle
            ty = dx * sin_angle + dy * cos_angle

            # Check if the point is inside the ellipse
            if (tx ** 2) / (major_axis / 2) ** 2 + (ty ** 2) / (minor_axis / 2) ** 2 > 1:  # True if outside ellipse.
                is_concentric = False
                break

    if is_concentric:
        # We will now remove the biggest contour.
        # Since contours are saved in a strange manner, we do some technical conversions to tuples.
        contours_tuples = [tuple(map(tuple, contour.reshape(-1, 2))) for contour in good_contours]
        contour_to_remove_tuple = tuple(map(tuple, biggest.reshape(-1, 2)))

        # Now actually remove the biggest contour.
        good_contours = [contour for contour, contour_tuple in zip(good_contours, contours_tuples)
                         if contour_tuple != contour_to_remove_tuple]

    return good_contours


def filter_contours_area(good_contours, area_ratio_threshold=0.30):
    """ This method will filter out the small contours, in the sense that the contours with the smallest area will
    be removed. This is to make sure that if, for some reason, the body of the salamander is separated with for
    example a paw or the head or the tail of the salamander, then we will remove these small parts and only focus
    on the body of the salamander as we desire. The area_ratio_threshold parameter determines when something is
    too small."""

    if not good_contours:
        return [], None

    sorted_contours = sorted(good_contours, key=contour_area, reverse=True)
    num_contours = len(sorted_contours)

    # Only consider the top three biggest contours with respect to the area, since the salamander is a big shape.
    largest_area = contour_area(sorted_contours[0]) if num_contours > 0 else 0
    second_area = contour_area(sorted_contours[1]) if num_contours > 1 else 0
    third_area = contour_area(sorted_contours[2]) if num_contours > 2 else 0

    # We will only keep the contours that are not too small with respect to each other.
    good_contours = [sorted_contours[0]]
    # Compare second and third-largest contours with the largest contour.
    if num_contours > 1 and second_area / largest_area >= area_ratio_threshold:
        good_contours.append(sorted_contours[1])
    if num_contours > 2 and third_area / largest_area >= area_ratio_threshold:
        good_contours.append(sorted_contours[2])

    return good_contours, sorted_contours[0]


def select_best_contour(good_contours, central_x, central_y, biggest_contour):
    """ This method selects the best contour from the remaining good contours. The method will pick the most central
    contour since that is the most likely to be a salamander."""

    if len(good_contours) > 0:
        closest_contour = None
        value_closest_contour = float("inf")

        for contour in good_contours:
            value_contour = closest_point_to_center(good_contours, central_x, central_y)

            if value_contour < value_closest_contour:
                closest_contour = contour
                value_closest_contour = value_contour

    else:
        # If there are no good contours, then we just take the contour which has the largest contour.
        closest_contour = biggest_contour

    return closest_contour


"""
(b) Removing paws, head and tail to only remain with the belly of the salamander.
"""


def put_rectangle_mask_on_image(image, rectangle_coords):
    """ This method removes some parts of the image, using a rectangle mask."""

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Create a mask with the same dimensions as the image, initialized to black (0).
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw the rectangle on the mask.
    cv.fillPoly(mask, [rectangle_coords], (255, 255, 255))

    # Only keep the part of the original image that is in the mask.
    masked_image = cv.bitwise_and(image_gray, mask)
    mask = masked_image > 0
    output_image = np.zeros_like(image)
    output_image[mask] = image[mask]

    return output_image


def rotate_image(image, x1, y1, x2, y2):
    """ This method rotates the image such that the salamander.
    Either of two scenarios is given back as output, depending on how the salamanders body was positioned originally.
    1. The salamander lies parallel to the x-axis with the head to the right.
    2. The salamander lies parallel to the y-axis with the head to the top."""

    # The points describe a rectangle, now first make this rectangle bigger.
    factor = 0.1
    x1 = x1 - x1 * factor
    y1 = y1 - y1 * factor
    x2 = x2 + x2 * factor
    y2 = y2 + y2 * factor

    # Now generate an image that has a bit less background removed (since the rectangle is bigger).
    big_rectangle_coords = np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], dtype=np.int32)
    image_with_less_crop = put_rectangle_mask_on_image(image, big_rectangle_coords)

    # Now detect the body of the salamander.
    image_with_less_crop_gray = cv.cvtColor(image_with_less_crop, cv.COLOR_BGR2GRAY)
    contours, _ = cv.findContours(image_with_less_crop_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    salamander_contour = max(contours, key=cv.contourArea)

    # Draw an ellipse around the body of the salamander.
    ellipse = cv.fitEllipse(salamander_contour)
    center, axes, angle = ellipse

    # If the orientation of the ellipse is closer to the x-axis, then rotate such that the ellipse is parallel with
    # the x-axis. And same reasoning for the y-axis. But we make sure that (if the salamander was originally with the
    # head pointing to the top or to the right) the head is again pointing to the top or to the right.
    if angle < 45:  # Closer to the y-axis.
        rotation_angle = angle
        orientation = 'vertical'

    elif angle > 135:  # Closer to the y-axis.
        rotation_angle = - (180 - angle)
        orientation = 'vertical'

    else:  # Closer to the x-axis.
        rotation_angle = - (90 - angle)
        orientation = 'horizontal'

    # Do the rotation.
    rotation_matrix = cv.getRotationMatrix2D(center, rotation_angle, 1.0)
    (h, w) = image.shape[:2]
    rotated_image = cv.warpAffine(image, rotation_matrix, (w, h))

    return rotated_image, orientation, center, rotation_angle


def contour_to_points(contour):
    """ This method will transform the contour in to a new datatype that is more suitable for further progress."""

    contour_points = []
    for big_list in contour:
        for point in big_list:
            contour_points.append(point)
    contour_points = np.array([contour_points], np.int32)

    return contour_points


def find_the_belly(best_contour, image):
    """ This method will try to crop the image even more, such that we can only see the belly of the salamander.
    First we will try to do detect the belly, but the salamander can be a bit rotated, that is why we need to rotate
    it back to a standardised pose (we do this by using the first try of finding the belly).
    Finally, we can try to find the belly again.

    INPUT: image is the isolated image.
    INPUT: best_contour is the contour surrounding this isolated image. """

    output_image = None
    orientation = None
    center = None
    angle = None
    counter = 0

    if best_contour is None:
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        best_contour, _ = cv.findContours(image_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        best_contour = max(best_contour, key=cv.contourArea)

    while counter <= 1:
        # First loop: detect the orientation of the salamander.
        # Second loop: cut of the head, paws and tail of the salamander.

        if counter == 0:
            contour_points = contour_to_points(best_contour)
        else:
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            best_contour, _ = cv.findContours(image_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            best_contour = max(best_contour, key=cv.contourArea)
            contour_points = contour_to_points(best_contour)

        if counter == 1:  # Cutting of the head of the salamander.
            if orientation == 'horizontal':
                x, y, w, h = cv.boundingRect(best_contour)
                cut_of_range = int(w * 0.2)  # Cutting of a percentage of the right side of the image.
                image[y:y + h, x + w - cut_of_range:x + w] = (0, 0, 0)

            if orientation == 'vertical':
                x, y, w, h = cv.boundingRect(best_contour)
                cut_of_range = int(h * 0.2)  # Cutting of a percentage of the top side of the image.
                image[y:y + cut_of_range, x:x + w] = (0, 0, 0)

            # Now again, find the contour surrounding the salamander (without the head hopefully).
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            best_contour, _ = cv.findContours(image_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            best_contour = max(best_contour, key=cv.contourArea)
            contour_points = contour_to_points(best_contour)

        # Find a rectangle inside the contour.
        rectangle = lir.lir(contour_points)

        (x1, y1) = lir.pt1(rectangle)  # Upper left point.
        (x2, y2) = lir.pt2(rectangle)  # Lower right point.

        extra = 0.05 * x1  # Originally, extra had a value of 20 but a more dependent factor seemed better.
        # Here, we try to enlarge the rectangle a bit to make sure that most of the belly is enclosed in the rectangle.
        if orientation is None:
            x1 = x1 - extra
            y1 = y1 - extra
            x2 = x2 + extra
            y2 = y2 + extra
        elif orientation == 'vertical':
            x1 = x1 - extra
            y1 = y1 + extra
            x2 = x2 + extra
            y2 = y2 - extra
        elif orientation == 'horizontal':
            x1 = x1 + extra
            y1 = y1 - extra
            x2 = x2 - extra
            y2 = y2 + extra

        rectangle_coords = np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], dtype=np.int32)

        if counter == 1:
            output_image = put_rectangle_mask_on_image(image, rectangle_coords)

        if counter == 0:
            image, orientation, center, angle = rotate_image(image, x1, y1, x2, y2)

        counter += 1

    # Rotate the image back to its original position, this is critical for other methods relying on isolate_salamander.
    rotation_matrix = cv.getRotationMatrix2D(center, - angle, 1.0)
    (h, w) = image.shape[:2]
    final_image = cv.warpAffine(output_image, rotation_matrix, (w, h))

    return final_image


def draw_best_contour(best_contour, image):
    """ This method will make a mask from the best contour and draw it on the original image."""

    # Create a mask for the selected contour
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.drawContours(mask, [best_contour], -1, [255], thickness=cv.FILLED)

    # Apply additional morphological operations like before.
    kernel = np.ones((15, 15), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # Apply the mask to the original image
    image = cv.bitwise_and(image, image, mask=mask)

    return image


"""
2. Isolating the belly of the salamander, using Pose Estimation.
"""


def select_useful_coordinates_from_pose_estimation(pose_estimation_dict):
    """ This method will select the shoulder, pelvis and spine coordinates, if the confidence is high enough. """

    useful_coordinates = dict()
    useful_names = ['spine_lowest', 'spine_low', 'spine_middle', 'spine_high', 'spine_highest', 'left_shoulder',
                    'right_shoulder', 'left_pelvis', 'right_pelvis']

    # First select the points with the highest confidence.
    for name, info in sorted(list(pose_estimation_dict.items()), key=lambda x: x[1][2], reverse=True):

        check_close = False
        if name in useful_names and pose_estimation_dict[name][2] >= pose_estimation_confidence:

            # If two points are too close together, we remove the point with the lowest confidence.
            for name2 in useful_coordinates:
                if math.dist(pose_estimation_dict[name][:2], useful_coordinates[name2][:2]) < 5:
                    check_close = True
                    break

            if not check_close:
                useful_coordinates[name] = pose_estimation_dict[name]

    assert_coordinates_from_pose_estimation(useful_coordinates)

    # Now we will do an additional check to make sure that the pelvis and shoulder are really at the right location.
    # We want to avoid that for example left_shoulder is located where left_pelvis should be.
    # We realise this by checking if the distance between for example left_shoulder and spine_highest is less than
    # the distance between left_shoulder and spine_lowest, as it should be. And these two always exist.
    if 'left_shoulder' in useful_coordinates.keys():
        distance1 = math.dist(pose_estimation_dict['left_shoulder'][:2], pose_estimation_dict['spine_highest'][:2])
        distance2 = math.dist(pose_estimation_dict['left_shoulder'][:2], pose_estimation_dict['spine_lowest'][:2])
        if distance1 > distance2:
            del useful_coordinates['left_shoulder']

    if 'right_shoulder' in useful_coordinates.keys():
        distance1 = math.dist(pose_estimation_dict['right_shoulder'][:2], pose_estimation_dict['spine_highest'][:2])
        distance2 = math.dist(pose_estimation_dict['right_shoulder'][:2], pose_estimation_dict['spine_lowest'][:2])
        if distance1 > distance2:
            del useful_coordinates['right_shoulder']

    if 'left_pelvis' in useful_coordinates.keys():
        distance1 = math.dist(pose_estimation_dict['left_pelvis'][:2], pose_estimation_dict['spine_highest'][:2])
        distance2 = math.dist(pose_estimation_dict['left_pelvis'][:2], pose_estimation_dict['spine_lowest'][:2])
        if distance1 < distance2:
            del useful_coordinates['left_pelvis']

    if 'right_pelvis' in useful_coordinates.keys():
        distance1 = math.dist(pose_estimation_dict['right_pelvis'][:2], pose_estimation_dict['spine_highest'][:2])
        distance2 = math.dist(pose_estimation_dict['right_pelvis'][:2], pose_estimation_dict['spine_lowest'][:2])
        if distance1 < distance2:
            del useful_coordinates['right_pelvis']

    assert_coordinates_from_pose_estimation(useful_coordinates)

    return useful_coordinates


def assert_coordinates_from_pose_estimation(coordinates):
    if ('left_pelvis' not in coordinates) and ('right_pelvis' not in coordinates):
        assert False, 'No left pelvis and right pelvis coordinates were detected.'
    if ('left_shoulder' not in coordinates) and ('right_shoulder' not in coordinates):
        assert False, 'No left shoulder and right shoulder coordinates were detected.'
    if ('spine_highest' not in coordinates) or ('spine_lowest' not in coordinates):
        assert False, 'No spine highest and lowest coordinates were detected.'


def find_torso(pose_estimation_dict):
    """ This method tries to find the shape of the torso of the salamander via interpolation. This enables us to
    also detect torsos that are curved."""

    # If spine_low, spine_middle and spine_high are not detected, few_spine_detected will be True, and we will need to
    # change our strategy a little bit.
    few_spine_detected: bool = False

    pose_estimation_dict = select_useful_coordinates_from_pose_estimation(pose_estimation_dict)

    # Check if we have enough points on the spine or if we need to change our strategy a bit.
    test_list = []
    for point in pose_estimation_dict:
        if point == 'spine_high' or point == 'spine_middle' or point == 'spine_low':
            test_list.append(point)
    if len(test_list) == 0:
        warnings.warn("Warning...........No points on the spine were detected. Results could be suboptimal.")
        few_spine_detected = True

    # The standard approach. Compute distance and angles (see our paper for the reason).
    coords_with_distance = compute_distances_for_torso(pose_estimation_dict, few_spine_detected)
    coords_with_angle = compute_angle_for_torso(coords_with_distance, pose_estimation_dict, few_spine_detected)

    points = find_coordinates_on_torso(coords_with_distance, coords_with_angle, pose_estimation_dict)

    # Interpolation:
    x = points[:, 0]
    y = points[:, 1]

    tck, u = splprep([x, y], s=0, per=True)

    return tck, points


def calculate_rico_for_torso(point1: Tuple[float | int, float | int], point2: Tuple[float | int, float | int]):
    """ This method calculates the rico, given two points. The rico is based on the horizontal axis."""

    if point1[0] == point2[0]:
        return 5000  # approximates infinity for our purposes.

    return (point2[1] - point1[1]) / (point2[0] - point1[0])


def calculate_angle_for_torso(rico1: float, rico2: float):
    """ This method will calculate the angle in degrees between two rico's. This angle lies
    in the interval [-180, 180]."""

    angle = 360 - math.degrees(math.atan((rico1 - rico2) / (1 + rico1 * rico2)))

    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360

    return angle


def compute_distances_for_torso(pose_estimation_dict, few_spine_detected: bool) -> list[Tuple[str, float | int]]:
    """ This method will calculate the distances between the detected point by the pose estimation and the points
    at the torso of the salamander (which are not detected by the pose estimation). """

    distances = []
    scaler = 1.2

    if 'left_shoulder' not in pose_estimation_dict:  # Buffer for when some parts are not detected by pose estimation.
        shoulder_distance = math.dist(pose_estimation_dict['right_shoulder'][:2],
                                      pose_estimation_dict['spine_highest'][:2])

    elif 'right_shoulder' not in pose_estimation_dict:
        shoulder_distance = math.dist(pose_estimation_dict['left_shoulder'][:2],
                                      pose_estimation_dict['spine_highest'][:2])

    else:  # Most frequent, usual, case when the pose estimation worked very well.
        left_shoulder_distance = math.dist(pose_estimation_dict['left_shoulder'][:2],
                                           pose_estimation_dict['spine_highest'][:2])
        right_shoulder_distance = math.dist(pose_estimation_dict['right_shoulder'][:2],
                                            pose_estimation_dict['spine_highest'][:2])
        shoulder_distance = np.average([left_shoulder_distance, right_shoulder_distance])

        # Enlarge the distance a bit such that we make sure the whole belly (in particular the whole torso) is captured.
        shoulder_distance = shoulder_distance * scaler

    if 'left_pelvis' not in pose_estimation_dict:  # Again a buffer for when things were not detected.
        pelvis_distance = math.dist(pose_estimation_dict['right_pelvis'][:2],
                                    pose_estimation_dict['spine_lowest'][:2])
    elif 'right_pelvis' not in pose_estimation_dict:
        pelvis_distance = math.dist(pose_estimation_dict['left_pelvis'][:2],
                                    pose_estimation_dict['spine_lowest'][:2])
    else:  # Usual case when everything goes well.
        left_pelvis_distance = math.dist(pose_estimation_dict['left_pelvis'][:2],
                                         pose_estimation_dict['spine_lowest'][:2])
        right_pelvis_distance = math.dist(pose_estimation_dict['right_pelvis'][:2],
                                          pose_estimation_dict['spine_lowest'][:2])
        pelvis_distance = np.average([left_pelvis_distance, right_pelvis_distance])

        # Enlarge the distance a bit such that we make sure the whole belly is captured.
        pelvis_distance = pelvis_distance * scaler

    if few_spine_detected:  # This changes our strategy a bit.
        distances.append(('pelvis', pelvis_distance))
        distances.append(('shoulder', shoulder_distance))

        return distances

    # Find smallest of both distances.
    if pelvis_distance >= shoulder_distance:
        delta_pelvis_shoulder = pelvis_distance - shoulder_distance
        smallest = shoulder_distance

    else:
        delta_pelvis_shoulder = shoulder_distance - pelvis_distance
        smallest = pelvis_distance

    temp_list = []
    for point in pose_estimation_dict:
        if point == 'spine_high' or point == 'spine_middle' or point == 'spine_low':
            temp_list.append(point)

    # Calculate the extra distance (that we add every step) necessary for a lineair interpolation.
    extra_distance = delta_pelvis_shoulder / (len(temp_list) + 2)
    factor = 1

    # Calculate the distances with the lineair interpolation.
    if smallest == pelvis_distance:

        distances.append(('pelvis', pelvis_distance))
        if 'spine_low' in temp_list:
            distances.append(('spine_low', pelvis_distance + factor * extra_distance))
            factor += 1
        if 'spine_middle' in temp_list:
            distances.append(('spine_middle', pelvis_distance + factor * extra_distance))
            factor += 1
        if 'spine_high' in temp_list:
            distances.append(('spine_high', pelvis_distance + factor * extra_distance))
            factor += 1
        distances.append(('shoulder', shoulder_distance))

    else:

        distances.append(('pelvis', pelvis_distance))
        if 'spine_low' in temp_list:
            distances.append(('spine_low', pelvis_distance - factor * extra_distance))
            factor += 1
        if 'spine_middle' in temp_list:
            distances.append(('spine_middle', pelvis_distance - factor * extra_distance))
            factor += 1
        if 'spine_high' in temp_list:
            distances.append(('spine_high', pelvis_distance - factor * extra_distance))
            factor += 1
        distances.append(('shoulder', shoulder_distance))

    return distances


def compute_angle_for_torso(coords_with_distance, pose_estimation_dict, few_spine_detected: bool) -> (
        list)[Tuple[str, float | int]]:
    """ This method will calculate the angle needed for a perpendicular line, starting at the detected point
    and ending at the torso of the salamander."""

    coords_with_angle = []

    # If spine_low, spine_middle and spine_high are not detected, then we need to change te strategy a bit.
    if few_spine_detected:

        if ('left_pelvis' in pose_estimation_dict) and ('right_pelvis' in pose_estimation_dict):
            point1 = pose_estimation_dict['left_pelvis'][:2]
            point2 = pose_estimation_dict['right_pelvis'][:2]
        elif 'left_pelvis' in pose_estimation_dict:
            point1 = pose_estimation_dict['left_pelvis'][:2]
            point2 = pose_estimation_dict['spine_lowest'][:2]
        else:
            point1 = pose_estimation_dict['right_pelvis'][:2]
            point2 = pose_estimation_dict['spine_lowest'][:2]

        rico_pelvis = calculate_rico_for_torso(point1, point2)
        angle_pelvis = calculate_angle_for_torso(0, rico_pelvis)

        coords_with_angle.append(('pelvis', angle_pelvis))

        if ('left_shoulder' in pose_estimation_dict) and ('right_shoulder' in pose_estimation_dict):
            point1 = pose_estimation_dict['left_shoulder'][:2]
            point2 = pose_estimation_dict['right_shoulder'][:2]
        elif 'left_shoulder' in pose_estimation_dict:
            point1 = pose_estimation_dict['left_shoulder'][:2]
            point2 = pose_estimation_dict['spine_highest'][:2]
        else:
            point1 = pose_estimation_dict['right_shoulder'][:2]
            point2 = pose_estimation_dict['spine_highest'][:2]

        rico_shoulder = calculate_rico_for_torso(point1, point2)
        angle_shoulder = calculate_angle_for_torso(0, rico_shoulder)

        coords_with_angle.append(('shoulder', angle_shoulder))

        return coords_with_angle

    # Otherwise we continue in the standard way.
    rico_list = []
    rico_pelvis = 0
    rico_shoulder = 0

    # Calculate the slope coefficient between every two points.
    for first, second in zip(coords_with_distance, coords_with_distance[1:]):
        first = first[0]
        second = second[0]

        if first == 'shoulder':
            break

        elif first == 'pelvis':
            first_coordinate = pose_estimation_dict['spine_lowest'][:2]

        else:
            first_coordinate = pose_estimation_dict[first][:2]

        if second == 'shoulder':
            second_coordinate = pose_estimation_dict['spine_highest'][:2]

        else:
            second_coordinate = pose_estimation_dict[second][:2]

        rico = calculate_rico_for_torso(first_coordinate, second_coordinate)

        rico_list.append((first, second, rico))

    # We need to solve an issue when some parts of the pelvis and/or shoulder are not detected well enough by the
    # Pose Estimation.
    need_to_solve_issue_pelvis = False
    need_to_solve_issue_shoulder = False

    # Try to calculate angles for the pelvis and shoulder.
    if ('left_pelvis' in pose_estimation_dict) and ('right_pelvis' in pose_estimation_dict):
        rico_pelvis = calculate_rico_for_torso(pose_estimation_dict['left_pelvis'][:2],
                                               pose_estimation_dict['right_pelvis'][:2])
        angle_pelvis = calculate_angle_for_torso(0, rico_pelvis)

        coords_with_angle.append(('pelvis', angle_pelvis))
        current_angle = angle_pelvis

    else:
        need_to_solve_issue_pelvis = True
        angle_pelvis = None
        current_angle = None

    if ('left_shoulder' in pose_estimation_dict) and ('right_shoulder' in pose_estimation_dict):
        rico_shoulder = calculate_rico_for_torso(pose_estimation_dict['left_shoulder'][:2],
                                                 pose_estimation_dict['right_shoulder'][:2])
        angle_shoulder = calculate_angle_for_torso(0, rico_shoulder)

    else:
        need_to_solve_issue_shoulder = True
        angle_shoulder = None

    counter = 1

    # Try to calculate the angles for the points on the spine.
    for first, second in zip(rico_list, rico_list[1:]):

        angle = calculate_angle_for_torso(first[2], second[2])

        if counter == 1 and need_to_solve_issue_pelvis:
            # We cannot compute the angle on the pelvis, so we try something artificial by using the next one and
            # turning it 90 degrees.

            if rico_list[0][2] != 0:
                rico_pelvis = - 1 / rico_list[0][2]
            else:
                rico_pelvis = np.sign(rico_shoulder) * 5000

            angle_pelvis = calculate_angle_for_torso(0, rico_pelvis)

            coords_with_angle.append(('pelvis', angle_pelvis))
            current_angle = angle_pelvis

        # Add the extra angle to take the curvature of the belly in to account.
        # These steps are technical but more information can be found in the paper.
        coords_with_angle.append((first[1], current_angle + angle))
        current_angle = current_angle + angle

        if second[1] == 'shoulder' and need_to_solve_issue_shoulder:

            if rico_list[-1][2] != 0:
                rico_shoulder = -1 / rico_list[-1][2]
            else:
                rico_shoulder = np.sign(rico_pelvis) * 5000

            angle_shoulder = calculate_angle_for_torso(0, rico_shoulder)

            coords_with_angle.append(('shoulder', angle_shoulder))

        counter += 1

    if not need_to_solve_issue_shoulder:
        coords_with_angle.append(('shoulder', angle_shoulder))

    return coords_with_angle


def ccw(A, B, C):
    """ Helper function for intersect."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    """ Return true if line segments AB and CD intersect"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def find_coordinates_on_torso(coords_with_distance, coords_with_angle, pose_estimation_dict):
    """ This method will find the coordinates on the torso of the salamander, using the distances and angles computed
    earlier."""

    # Split up the coordinates in two list, both containing points on one side of the torso.
    lower_coordinates = []
    upper_coordinates = []

    for name, distance in coords_with_distance:
        angle = None

        for name2, angle2 in coords_with_angle:
            if name == name2:
                angle = angle2
                break

        if name == 'pelvis':
            base_coordinate = pose_estimation_dict['spine_lowest'][:2]

        elif name == 'shoulder':
            base_coordinate = pose_estimation_dict['spine_highest'][:2]

        else:
            base_coordinate = pose_estimation_dict[name][:2]

        # Calculate the position of the points on the torso, using polar coordinates.
        lower_coordinate_x = int(base_coordinate[0] - distance * math.cos(math.radians(angle)))
        lower_coordinate_y = int(base_coordinate[1] - distance * math.sin(math.radians(angle)))

        upper_coordinate_x = int(base_coordinate[0] + distance * math.cos(math.radians(angle)))
        upper_coordinate_y = int(base_coordinate[1] + distance * math.sin(math.radians(angle)))

        lower_coordinates.append([lower_coordinate_x, lower_coordinate_y])
        upper_coordinates.append([upper_coordinate_x, upper_coordinate_y])

    # Reverse the order of the coordinates of the list such that if we put both lists together, we will get the points
    # in consecutive order. We need this to be able to interpolate a curve through the points and this curve must
    # approximate the shape of the belly. So we really want to have the points in a nice consecutive order.
    lower_coordinates.reverse()

    # We also, as an extra check, make sure the curve does not intersect with itself.
    # This does happen sometimes, and we do not really know why, but this fixes that problem.
    if intersect(upper_coordinates[-2], upper_coordinates[-1], lower_coordinates[0], lower_coordinates[1]):
        temp = upper_coordinates[-1]
        upper_coordinates[-1] = lower_coordinates[0]
        lower_coordinates[0] = temp

    coordinates = upper_coordinates + lower_coordinates

    # Add starting point also to the end of the list since the start and end point is the same.
    coordinates.append(coordinates[0])

    return np.array(coordinates)


def remove_everything_outside_curve(image, tck):
    """ This function converts everything outside the given curve, thus only the belly will remain with black
    background."""

    #  Make the interpolation curve through the calculated points on the torso.
    u_dense = np.linspace(0, 1, 500)
    x_dense, y_dense = splev(u_dense, tck)

    x_dense = np.clip(x_dense, 0, image.shape[1] - 1).astype(np.int32)
    y_dense = np.clip(y_dense, 0, image.shape[0] - 1).astype(np.int32)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Only remain with the part inside this curve (=belly).
    contour = np.stack((x_dense, y_dense), axis=1)
    cv.fillPoly(mask, [contour], color=[255])

    output = cv.bitwise_and(image, image, mask=mask)

    return output


def resize(image, width=None, height=750, inter=cv.INTER_AREA):
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


def crop_image(image: np.array, coordinates_pose: Dict[str, Tuple[int, int, float]] | None = None,
               is_background_removed: bool = False, pose_estimation_evaluation: int = 0):
    # Only gets used in facade.py
    """ This method will literally crop the image. So we remove the non-interesting background.
    This removes everything except (the belly) of the salamander.

    Further documentation: see documentation from isolate_salamander. """

    image = isolate_salamander(image, coordinates_pose, is_background_removed, pose_estimation_evaluation)

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    x, y, w, h = cv.boundingRect(image_gray)

    # Now crop the image to the bounding box.
    cropped_image = image[y: y + h, x: x + w]

    return cropped_image
