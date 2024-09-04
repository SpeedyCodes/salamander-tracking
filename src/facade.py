import numpy as np
from src.pose_estimation import estimate_pose_from_image, CoordinateTransformer
from src.dot_detection import dot_detect_haar
from src.preprocessing import normalise_coordinates, crop_image
from src.pattern_matching import compare_dot_patterns
from server.database_interface import get_individuals_coords

"""
The Facade abstracts away the intricacies of salamander recognition and matching to make it more accessible.
Internally, it defines from a high level the pipeline of operations that need to be performed to match a salamander.

"""


class CoordinateExtractor:

    def __init__(self, image: np.ndarray):
        self.coordinates_pose = None
        self.pose_estimation_success = None
        self.image = image

    def crop_image_to_belly(self) -> np.ndarray:
        """
        :return: A cropped version of the image. In this cropped version, we only see the belly of the salamander,
        or the whole salamander if so wished.
        """

        # First, try to use pose estimation.
        pose_estimation_evaluation = 2
        is_background_removed = False
        no_background = None
        cropped_image = None

        self.coordinates_pose, self.pose_estimation_success = estimate_pose_from_image(self.image)

        if not self.pose_estimation_success:
            # Try to crop the image such that it is smaller.
            pose_estimation_evaluation -= 1

        if pose_estimation_evaluation == 1:
            try:
                # Remove the background.
                no_background = crop_image(self.image, self.coordinates_pose, False,
                                           pose_estimation_evaluation)
            except AssertionError:
                pose_estimation_evaluation = 0
            else:
                is_background_removed = True
                pose_estimation_evaluation += 1

        if pose_estimation_evaluation == 2 and is_background_removed:
            # Try pose estimation again on smaller image.
            self.coordinates_pose, self.pose_estimation_success = estimate_pose_from_image(no_background)

            if not self.pose_estimation_success:
                pose_estimation_evaluation = 0

        while pose_estimation_evaluation == 0 or pose_estimation_evaluation == 2:
            # If equal to 2, then we use the pose estimation based method, otherwise we use the old method.

            if is_background_removed:
                current_image = no_background
            else:
                current_image = self.image

            try:
                cropped_image = crop_image(current_image, self.coordinates_pose, is_background_removed,
                                           pose_estimation_evaluation)
            except AssertionError:
                if pose_estimation_evaluation == 2:
                    pose_estimation_evaluation = 0
                else:
                    cropped_image = current_image
                    pose_estimation_evaluation = None  # To break out the while loop.
            else:
                pose_estimation_evaluation = None  # To break out the while loop.

        return cropped_image

    def extract(self) -> set[tuple[float, float]]:
        """
        :return: The canonical representation of the coordinates of the dots on the salamander's skin in the image.
        """
        cropped_image = self.crop_image_to_belly()
        list_haar_cascade = dot_detect_haar(cropped_image)
        list_coordinates = normalise_coordinates(list_haar_cascade, cropped_image.shape)
        if self.pose_estimation_success:
            coordinate_transformer = CoordinateTransformer(self.coordinates_pose)
            list_coordinates = set([coordinate_transformer.transform(*coordinate) for coordinate in list_coordinates])

        return list_coordinates


def image_to_canonical_representation(image: np.ndarray) -> set[tuple[float, float]]:
    """
    Takes in an image and returns the canonical (normalized) representation of the coordinates.
    :param:  image
    :return: The canonical representation of the coordinates of the dots on the salamander's skin in the image.
    """
    coordinate_extractor = CoordinateExtractor(image)

    return coordinate_extractor.extract()


def match_canonical_representation_to_database(canonical_representation: set[tuple[float, float]], candidates_number) -> str | None:
    """
    Matches the canonical representation of the image to an entry in the database
    :param canonical_representation:
    :return: The name of the salamander in the database
    """
    database = get_individuals_coords()
    scores = compare_dot_patterns(canonical_representation, database)
    actual_candidates_number = min(candidates_number, len(scores))
    return scores[:actual_candidates_number]
