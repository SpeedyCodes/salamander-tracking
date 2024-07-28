"""
Rune De Coninck

This document will use an algorithm [Edward J. Groth 1986: A pattern-matching algorithm for two-dimensional coordinate
lists], [Journal of Applied Ecology 2005: An astronomical pattern-matching algorithm for computer-aided identification
of whale sharks Rhincodon typus] to compare a list of coordinates (which represents dots of the salamanders).
Thus, the goal is to match or not match certain patterns of two salamanders and thus, tell if we are working with the
same salamander or not.

This is done in several steps:
0. Loading in the list of coordinates/points.
1. Check if the number of coordinates/points is approximately equal.
2. Selecting the points to be matched.
3. [NOT DONE YET] Generating lists of triangles.
4. [NOT DONE YET] Filtering the triangles.
5. [NOT DONE YET] Matching the triangles.
6. [NOT DONE YET] Reducing the number of false matches.
7. [NOT DONE YET] Assigning matched points.
8. [NOT DONE YET] Protecting against spurious assignments.
"""

from dot_detection.dot_detect_haar import dot_detect_haar
from Convert_image_to_stips import crop_detailed_image_small_size
import numpy as np
import math
import itertools


def compare_dot_patterns(image: np.ndarray, database: set[tuple[np.ndarray, str]], tol: float = 0.001):
    """ This function will include a bunch of other functions ...

    INPUT: image denotes the unknown image, this is the image we want to compare with our database.
    INPUT: database is a set of tuples (image, name) that represents all the items of our database.
    INPUT: the tolerance tol is a parameter that the user is free to set. If detected dots have distance less than
    a scaler multiplied by tol, then we consider these dots as one. (We remove one of the two dots, this ensures that
    the numerical computations are stable).
    OUTPUT: """

    # First, crop the image to the essential part.
    image = crop_detailed_image_small_size(image, reverse_colors=False)

    # Second, load in the list of coordinates of the unknown image and select the points to be used.
    list_coordinates = load_in_coordinates(image)
    list_coordinates = select_points_to_be_matched(list_coordinates, tol=3*tol)

    # Then, transform the database of images to a database of lists of coordinates with a name tag.
    database_of_coordinates = set()  # type = set[ list[ set[tuple[float, float] ], str ].
    # Essentially this is of the form set[ list[ list_of_coordinates, name_tag ] ]

    for image_database, name_image_database in database:
        image_database = crop_detailed_image_small_size(image_database, reverse_colors=False)
        list_coordinates_image_database = load_in_coordinates(image_database)
        list_coordinates_image_database = select_points_to_be_matched(list_coordinates_image_database, tol=3*tol)
        database_of_coordinates.add([list_coordinates_image_database, name_image_database])

    # Now start the matching procedure.
    for list_coordinates_image_database, name_image_database in database_of_coordinates:

        # Check the number of detected points.
        if not check_number_of_points(list_coordinates, list_coordinates_image_database):
            continue  # This is not a match, go to the next pattern in the database.

    return None


""" STEP 0: Loading in the list of coordinates/points. """


def calculate_centroid_of_rectangle(x, y, width, height):
    """ This method calculates the centroid coordinates of a rectangle, given that the origin is at the top left
    of the image and given the parameters (x, y, width, height),
    where (x, y) is the coordinate of the top left corner of the rectangle and width is the horizontal length of the
    rectangle and height is the vertical length of the rectangle."""

    return (x + width / 2), (y + height / 2)


def normalisation_of_coordinates(x, y, width_image, height_image):
    """ This method normalizes the coordinates of a rectangle to the [0, 1] interval,
    given that the origin is at the top left."""

    return x/width_image, y/height_image


def transform_origin(x, y):
    """ This method transforms the coordinates of a rectangle to a new origin, located at the bottom left of the
    image, not the top left of the image. We assume that the coordinates are already normalised."""

    return x, 1 - y


def load_in_coordinates(image) -> set[tuple[float, float]]:
    """ This method loads the coordinates of the dots which are detected in the haar cascade.
    We extract the centroid of this input and also normalize the coordinates to the [0, 1] interval.
    Furthermore, we use a new origin, located at the bottom left of the image."""

    width_image, height_image = image.shape

    list_coordinates = set()

    # list_of_coordinates returns a list containing 4-tuples of the form (x, y, width, height).
    # This rectangle surrounds the dot of the salamander. The origin of the used coordinate system is at the top left
    # of the image. (x, y) denotes the upper left coordinate of the rectangle. Width and height respectively denote
    # the length of the side of the rectangle, starting from the (x, y) coordinate, in x and y direction.
    list_haar_cascade = dot_detect_haar(image)

    for (x, y, w, h) in list_haar_cascade:
        # First, we try to detect the center of the rectangle in the given coordinate system.
        x_centroid, y_centroid = calculate_centroid_of_rectangle(x, y, w, h)

        # Second, we normalize the coordinates and use a new coordinate system (with the origin at the bottom left
        # of the image).
        x_centroid, y_centroid = normalisation_of_coordinates(x_centroid, y_centroid, width_image, height_image)
        x_centroid, y_centroid = transform_origin(x_centroid, y_centroid)

        list_coordinates.add((x_centroid, y_centroid))

    return list_coordinates


""" STEP 1: Selecting the points to be matched."""


def select_points_to_be_matched(list_coordinates, tol: float):
    """ This method will select the points that we can use for the remainder of the algorithm.
    If two points are closer than a distance tol of each other, we will remove one of the two points and consider them
    to be the same point. This ensures good numerical calculations."""

    selected_points = set()

    for point in list_coordinates:
        # Check if the point is too close to any of the already selected points.
        is_close = False
        for selected in selected_points:
            distance = math.dist(point, selected)
            if distance < tol:
                is_close = True
                break
        # If not close, add the point to the selected points list.
        if not is_close:
            selected_points.add(point)

    return selected_points


""" STEP 2: Check if the number of coordinates/points is approximately equal. """


def check_number_of_points(list_coordinates1, list_coordinates2, tol: int = 5):
    """ This method returns True if the difference of number of points is less than the tolerance and False otherwise.
    In this way, a salamander with a lot of points won't get matched with a salamander with few points.
    Furthermore, this ensures that the lists of coordinates are approximately of equal size. From the literature,
    we know that this ensures good results."""

    return abs(len(list_coordinates1) - len(list_coordinates2)) <= tol


""" STEP 3: Generating lists of triangles."""


def generate_triangles(list_coordinates):
    """ This method will generate all possible triangles between every three distinct points in the list of coordinates.
    We return a big list of all the triangles and extra info such as ratio of sides, perimeter, orientation,
    different tolerances, cosine of a certain angle. Detailed info can be found in this code as comments."""

    list_triangles = set()
    NOG NIET AF
    # First create all possible triangles between every three points in the list of coordinates.
    triangles = []

    # Generate all the combinations of three distinct points.
    for point1, point2, point3 in itertools.combinations(list_coordinates, 3):
        # Calculate the lengths of the sides of the triangle
        side1 = math.dist(point1, point2)
        side2 = math.dist(point2, point3)
        side3 = math.dist(point3, point1)

        triangles.append(((point1, point2, point3), (side1, side2, side3)))

    return triangles
