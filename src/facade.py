from typing import Dict, Tuple
import numpy as np
from src.pose_estimation.estimate import estimate_pose_from_image
from src.dot_detection import dot_detect_haar
from src.preprocessing import crop_image, normalise_coordinates
from src.pattern_matching import compare_dot_patterns

"""
The Facade abstracts away the intricacies of salamander recognition and matching to make it more accessible.
Internally, it defines from a high level the pipeline of operations that need to be performed to match a salamander.

"""


def main_body_parts_to_coordinates(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Tuple[int, int, float]]]:
    """
    Takes in an image and returns the coordinates of the main body parts of the salamander.
    :param image:
    :return: The main body parts of the salamander.
    """

    h, w = image.shape[:2]

    # Case 1, the image is too big for the pose estimation.
    if h > 2500 or w > 2500:

        # Obtains the image without background, but still full salamander.
        image = crop_image(image, crop_to_belly=False)

        # Now try the pose estimation again on the smaller image.
        main_body_parts = estimate_pose_from_image(image)

    # Case 2, the image is small enough for the pose estimation.
    else:
        main_body_parts = estimate_pose_from_image(image)

    return image, main_body_parts


def image_to_canonical_representation(image: np.ndarray) -> set[tuple[float, float]]:
    """
    Takes in an image and returns the canonical representation of the image that is returned from
    main_body_parts_to_coordinates.
    :param:  The image that is returned from main_body_parts_to_coordinates
    :return: The canonical representation of the coordinates of the dots on the salamander's skin in the image
    """

    cropped_image = crop_image(image)  # Fully crops, only the belly of the salamander remains.
    list_haar_cascade = dot_detect_haar(image)
    list_coordinates = normalise_coordinates(list_haar_cascade, cropped_image.shape)

    return list_coordinates


def match_canonical_representation_to_database(canonical_representation: set[tuple[float, float]]) -> str | None:
    """
    Matches the canonical representation of the image to an entry in the database
    :param canonical_representation:
    :return: The name of the salamander in the database
    """
    database = []  # TODO
    scores = compare_dot_patterns(canonical_representation, database)
    if len(scores) == 0:
        return None
    return scores[1]
