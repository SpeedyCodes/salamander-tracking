""" Rune De Coninck

Given a picture, this document will try to isolate the salamander from the background.
Hence, it will return a new image, without minimal background and as much of the salamander.
We will commence by using color segmentation and morphological operations to detect the salamander.
After that, we will try to remove some parts of the background.
"""

import cv2 as cv
import numpy as np
import largestinteriorrectangle as lir


def isolate_salamander(image: np.array, is_old_and_new: bool = False) -> np.array:
    """ This function will include a bunch of other functions, it will return a black image with only the salamander.

    INPUT: numpy array image, this must! be in BGR format, this type of image can be obtained by wrapped_imread.
    INPUT: if is_old_and_new is True, it will return the old isolate salamander and the new cropped version of
    the belly. Otherwise, if false, then it will just return the new cropped version of the belly.
    OUTPUT: if is_old_and_new is True, then it will return two images:
    First the new cropped version of the belly, then the old isolate salamander.
    Otherwise, it will only return the new cropped version of the belly.
    In all cases, the output images are numpy array images."""

    # First, detect the salamander (and mask) with color segmentation, thus there will be some noise left.
    image_isolated_salamander_with_noise, mask = color_segmentation(image, ksize=51, lower_bound=[5, 50, 50],
                                                                    upper_bound=[35, 255, 255])

    # Second, try to filter out the noise based on the fact that the contour of the salamander is a big central object.
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

    # Try to find the belly of the salamander.
    image_belly = find_the_belly(best_contour, image_isolated_without_noise)
    if is_old_and_new:
        return image_isolated_without_noise, image_belly
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
    if len(biggest) == 4:
        raise Exception('Only contour found is the trivial one, please try again with a better image!')

    assert len(biggest) >= 5, 'No good contours found!'
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

    return rotated_image, orientation


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
    counter = 0

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
            image, orientation = rotate_image(image, x1, y1, x2, y2)

        counter += 1

    return output_image


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
