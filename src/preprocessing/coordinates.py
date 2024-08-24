
def normalise_coordinates(raw_coordinates, image_shape) -> set[tuple[float, float]]:
    """
    We extract the centroid of this input and also normalize the coordinates to the [0, 1] interval.
    Furthermore, we use a new origin, located at the bottom left of the image."""

    height_image = image_shape[0]
    width_image = image_shape[1]

    normalised_coordinates = set()

    # raw_coordinates is a list containing 4-tuples of the form (x, y, width, height).
    # This rectangle surrounds the dot of the salamander. The origin of the used coordinate system is at the top left
    # of the image. (x, y) denotes the upper left coordinate of the rectangle. Width and height respectively denote
    # the length of the side of the rectangle, starting from the (x, y) coordinate, in x and y direction.
    if len(raw_coordinates) == 0:
        return set()

    for (x, y, w, h) in raw_coordinates:
        # First, we try to detect the center of the rectangle in the given coordinate system.
        x_centroid, y_centroid = calculate_centroid_of_rectangle(x, y, w, h)

        # Second, we normalize the coordinates and use a new coordinate system (with the origin at the bottom left
        # of the image).
        x_centroid, y_centroid = normalisation_of_coordinates(x_centroid, y_centroid, width_image, height_image)
        x_centroid, y_centroid = transform_origin(x_centroid, y_centroid)

        normalised_coordinates.add((x_centroid, y_centroid))

    return normalised_coordinates


def transform_origin(x, y):
    """ This method transforms the coordinates of a rectangle to a new origin, located at the bottom left of the
    image, not the top left of the image. We assume that the coordinates are already normalised."""

    return x, 1 - y


def normalisation_of_coordinates(x, y, width_image, height_image):
    """ This method normalizes the coordinates of a rectangle to the [0, 1] interval,
    given that the origin is at the top left."""

    return x / width_image, y / height_image


def calculate_centroid_of_rectangle(x, y, width, height):
    """ This method calculates the centroid coordinates of a rectangle, given that the origin is at the top left
    of the image and given the parameters (x, y, width, height),
    where (x, y) is the coordinate of the top left corner of the rectangle and width is the horizontal length of the
    rectangle and height is the vertical length of the rectangle."""

    return (x + width / 2), (y + height / 2)
