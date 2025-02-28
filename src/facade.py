import numpy as np

from src.dot_detection.dot_detect_haar import draw_dots
from src.pose_estimation import estimate_pose_from_image, CoordinateTransformer, draw_pose
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
        self.pose_estimation_image = None
        self.cropped_image = None
        self.dot_detection_image = None
        self.straightened_dots_image = None

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
        self.pose_estimation_image = draw_pose(self.image, self.coordinates_pose)

        if not self.pose_estimation_success:
            # Try to crop the image such that it is smaller.
            pose_estimation_evaluation -= 1

        if pose_estimation_evaluation == 1:
            try:
                # Remove the background.
                no_background, _ = isolate_salamander(self.image, self.coordinates_pose, False,
                                                      pose_estimation_evaluation)
            except AssertionError:
                pose_estimation_evaluation = 0
            else:
                is_background_removed = True
                pose_estimation_evaluation += 1

        if pose_estimation_evaluation == 2 and is_background_removed:
            # Try pose estimation again on smaller image.
            self.coordinates_pose, self.pose_estimation_success = estimate_pose_from_image(no_background)
            self.pose_estimation_image = draw_pose(self.image, self.coordinates_pose)

            if not self.pose_estimation_success:
                pose_estimation_evaluation = 0

        while pose_estimation_evaluation == 0 or pose_estimation_evaluation == 2:
            # If equal to 2, then we use the pose estimation based method, otherwise we use the old method.

            if is_background_removed:
                current_image = no_background
            else:
                current_image = self.image

            try:
                isolate_image, few_spine_detected = isolate_salamander(current_image, self.coordinates_pose,
                                                                       is_background_removed,
                                                                       pose_estimation_evaluation)
            except AssertionError:
                if pose_estimation_evaluation == 2:
                    pose_estimation_evaluation = 0
                else:
                    isolate_image = current_image
                    pose_estimation_evaluation = None  # To break out the while loop.
                    image_quality = ImageQuality.BAD
            else:
                if pose_estimation_evaluation == 2:
                    if few_spine_detected:
                        image_quality = ImageQuality.MEDIUM
                    else:
                        image_quality = ImageQuality.GOOD
                else:
                    image_quality = ImageQuality.MEDIUM
                break

        return isolate_image, image_quality

    def extract(self) -> tuple[set[tuple[float, float]], ImageQuality]:
        """
        :return: The canonical representation of the coordinates of the dots on the salamander's skin in the image.
        """
        self.cropped_image, image_quality = self.crop_image_to_belly()
        list_haar_cascade = dot_detect_haar(self.cropped_image)

        self.dot_detection_image = draw_dots(self.cropped_image, list_haar_cascade)

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
            self.straightened_dots_image = coordinate_transformer.show_transformation(self.image)
        else:  # TODO check if the normalisation outside transform() doesn't need to happen even if pose estimation succeeds
            normalised = [
                normalisation_of_coordinates(*coordinate, self.cropped_image.shape[1], self.cropped_image.shape[0]) for
                coordinate in list_coordinates]

        return normalised, image_quality


def image_to_canonical_representation(image: np.ndarray) -> tuple[
        set[tuple[float, float]], ImageQuality, list[np.ndarray]]:
    """
    Takes in an image and returns the canonical (normalized) representation of the coordinates.
    :param:  image
    :return: The canonical representation of the coordinates of the dots on the salamander's skin in the image.
    :return: The quality of the image, this is either 'Good', 'Medium', or 'Bad'.
    """
    coordinate_extractor = CoordinateExtractor(image)
    coords, quality = coordinate_extractor.extract()
    return coords, quality, [coordinate_extractor.pose_estimation_image, coordinate_extractor.cropped_image,
                             coordinate_extractor.dot_detection_image, coordinate_extractor.straightened_dots_image]


def match_canonical_representation_to_database(canonical_representation: set[tuple[float, float]],
                                               candidates_number: int, location_id: int | None) -> str | None:
    """
    Matches the canonical representation of the image to an entry in the database
    :param canonical_representation: The canonical representation of the image
    :param candidates_number: The number of candidates to return
    :param location_id: The location of the salamander
    :return: The name of the salamander in the database
    """
    database = get_individuals_coords(location_id)
    scores = compare_dot_patterns(canonical_representation, database)
    actual_candidates_number = min(candidates_number, len(scores))
    return scores[:actual_candidates_number]
