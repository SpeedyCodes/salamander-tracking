import numpy as np
from src.pose_estimation import estimate_pose_from_image
from src.dot_detection import dot_detect_haar
from src.preprocessing import crop_image, normalise_coordinates
from src.pattern_matching import compare_dot_patterns

"""
The Facade abstracts away the intricacies of salamander recognition and matching to make it more accessible.
Internally, it defines from a high level the pipeline of operations that need to be performed to match a salamander.

"""

def image_to_canonical_representation(image: np.ndarray) -> set[tuple[float, float]]:
    """
    Takes in an image and returns the canonical representation of the image
    :param image:
    :return: The canonical representation of the coordinates of the dots on the salamander's skin in the image
    """


    cropped_image = crop_image(image)
    list_haar_cascade = dot_detect_haar(cropped_image)
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
