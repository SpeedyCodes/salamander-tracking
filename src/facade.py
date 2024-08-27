from typing import Dict, Tuple
import numpy as np
from src.pose_estimation.estimate import estimate_pose_from_image
from src.dot_detection import dot_detect_haar
from src.preprocessing import normalise_coordinates
from src.preprocessing.isolate_salamander import crop_image
from src.pattern_matching import compare_dot_patterns

"""
The Facade abstracts away the intricacies of salamander recognition and matching to make it more accessible.
Internally, it defines from a high level the pipeline of operations that need to be performed to match a salamander.

"""


def crop_image_to_belly(image: np.ndarray) -> np.ndarray:
    """
    :param image:
    :return: A cropped version of the image. In this cropped version, we only see the belly of the salamander,
    or the whole salamander if so wished.
    """

    # First, try to use pose estimation.
    pose_estimation_evaluation = 2
    is_background_removed = False
    no_background = None
    cropped_image = None

    coordinates_pose, succes = estimate_pose_from_image(image)

    if not succes:
        # Try to crop the image such that it is smaller.
        pose_estimation_evaluation -= 1

    if pose_estimation_evaluation == 1:
        try:
            # Remove the background.
            no_background = crop_image(image, coordinates_pose, False, pose_estimation_evaluation)
        except:
            pose_estimation_evaluation = 0
        else:
            is_background_removed = True
            pose_estimation_evaluation += 1

    if pose_estimation_evaluation == 2 and is_background_removed:
        # Try pose estimation again on smaller image.
        coordinates_pose, succes = estimate_pose_from_image(no_background)

        if not succes:
            pose_estimation_evaluation = 0

    if pose_estimation_evaluation == 2:

        if is_background_removed:

            try:
                cropped_image = crop_image(no_background, coordinates_pose, is_background_removed,
                                           pose_estimation_evaluation)
            except:
                pose_estimation_evaluation = 0

        else:
            try:
                cropped_image = crop_image(image, coordinates_pose, is_background_removed, pose_estimation_evaluation)
            except:
                pose_estimation_evaluation = 0

    if pose_estimation_evaluation == 0:

        if is_background_removed:
            try:
                cropped_image = crop_image(no_background, coordinates_pose, is_background_removed,
                                           pose_estimation_evaluation)
            except:
                cropped_image = no_background

        else:
            try:
                cropped_image = crop_image(image, coordinates_pose, is_background_removed, pose_estimation_evaluation)
            except:
                cropped_image = image

    return cropped_image


def image_to_canonical_representation(image: np.ndarray) -> set[tuple[float, float]]:
    """
    Takes in an image and returns the canonical (normalized) representation of the coordinates.
    :param:  image
    :return: The canonical representation of the coordinates of the dots on the salamander's skin in the image.
    """
    cropped_image = crop_image_to_belly(image)
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
