import numpy as np
from src.pose_estimation import estimate_pose_from_image, CoordinateTransformer
from src.dot_detection import dot_detect_haar
from src.preprocessing import isolate_salamander
from src.preprocessing.coordinates import calculate_centroid_of_rectangle, normalisation_of_coordinates
from src.pattern_matching import compare_dot_patterns
from server.database_interface import get_individuals_coords
from src.utils import ImageQuality

"""
The Facade abstracts away the intricacies of salamander recognition and matching to make it more accessible.
Internally, it defines from a high level the pipeline of operations that need to be performed to match a salamander.

"""


class CoordinateExtractor:

    def __init__(self, image: np.ndarray):
        self.coordinates_pose = None
        self.pose_estimation_success = None
        self.image = image

    def crop_image_to_belly(self) -> tuple[np.ndarray, ImageQuality]:
        """
        :return: A cropped version of the image. In this cropped version, we only see the belly of the salamander,
        or the whole salamander if so wished.
        :return: A parameter which can be GOOD, MEDIUM, or BAD, which denotes the quality of the image.
        """

        # First, try to use pose estimation.
        pose_estimation_evaluation = 2
        is_background_removed = False
        no_background = None
        isolate_image = None
        image_quality: ImageQuality = ImageQuality.GOOD

        self.coordinates_pose, self.pose_estimation_success = estimate_pose_from_image(self.image)

        if not self.pose_estimation_success:
            # Try to crop the image such that it is smaller.
            pose_estimation_evaluation -= 1

        if pose_estimation_evaluation == 1:
            try:
                # Remove the background.
                no_background = isolate_salamander(self.image, self.coordinates_pose, False,
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
                isolate_image = isolate_salamander(current_image, self.coordinates_pose, is_background_removed,
                                                   pose_estimation_evaluation)
            except AssertionError:
                if pose_estimation_evaluation == 2:
                    pose_estimation_evaluation = 0
                else:
                    isolate_image = current_image
                    pose_estimation_evaluation = None  # To break out the while loop.
                    image_quality = ImageQuality.BAD
            else:
                pose_estimation_evaluation = None  # To break out the while loop.
                if pose_estimation_evaluation == 2:
                    image_quality = ImageQuality.GOOD
                else:
                    image_quality = ImageQuality.MEDIUM

        return isolate_image, image_quality

    def extract(self) -> tuple[set[tuple[float, float]], ImageQuality]:
        """
        :return: The canonical representation of the coordinates of the dots on the salamander's skin in the image.
        """
        isolate_image, image_quality = self.crop_image_to_belly()
        list_haar_cascade = dot_detect_haar(isolate_image)

        if len(list_haar_cascade) < 3:
            # There are either no dots detected, which is very bad. Or there are too little dots detected to generate
            # triangles in the matching algorithm.
            image_quality = ImageQuality.BAD

        if image_quality == ImageQuality.BAD:
            # Then we do not need to do the following parts, we need to ask for a new image.
            return set(), image_quality

        list_coordinates = [calculate_centroid_of_rectangle(*coordinate) for coordinate in list_haar_cascade]
        if self.pose_estimation_success:
            coordinate_transformer = CoordinateTransformer(self.coordinates_pose)
            if coordinate_transformer.image_quality == ImageQuality.BAD:
                return set(), coordinate_transformer.image_quality
            image_quality = min(image_quality, coordinate_transformer.image_quality)
            normalised = set([coordinate_transformer.transform(*coordinate) for coordinate in list_coordinates])
        else:  # TODO check if the normalisation outside transform() doesn't need to happen even if pose estimation succeeds
            normalised = [normalisation_of_coordinates(*coordinate, isolate_image.shape[1], isolate_image.shape[0]) for
                          coordinate in list_coordinates]

        return normalised, image_quality


def image_to_canonical_representation(image: np.ndarray) -> tuple[set[tuple[float, float]], ImageQuality]:
    """
    Takes in an image and returns the canonical (normalized) representation of the coordinates.
    :param:  image
    :return: The canonical representation of the coordinates of the dots on the salamander's skin in the image.
    :return: The quality of the image, this is either 'Good', 'Medium', or 'Bad'.
    """
    coordinate_extractor = CoordinateExtractor(image)

    return coordinate_extractor.extract()


def match_canonical_representation_to_database(canonical_representation: set[tuple[float, float]],
                                               candidates_number) -> str | None:
    """
    Matches the canonical representation of the image to an entry in the database
    :param canonical_representation:
    :return: The name of the salamander in the database
    """
    database = get_individuals_coords()
    scores = compare_dot_patterns(canonical_representation, database)
    actual_candidates_number = min(candidates_number, len(scores))
    return scores[:actual_candidates_number]
