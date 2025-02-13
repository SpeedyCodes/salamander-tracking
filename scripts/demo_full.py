"""
Jesse Daems & Rune De Coninck
Full demo script for 05/09/2024

legend:
1. Pose Estimation
2. Isolate belly with Pose Estimation
3. Isolate belly with Color Segmentation
4. Haar Cascade
5. Compare dot patterns + full run
"""
from src.facade import image_to_canonical_representation
from src.pattern_matching import compare_dot_patterns, display_results
from src.dot_detection.dot_detect_haar import dot_detect_haar, draw_dots
from src.preprocessing.isolate_salamander import isolate_salamander, crop_image
from src.utils import wrapped_imread
from server.database_interface import get_individuals_coords, get_dataclass, get_file
from src.pose_estimation import estimate_pose_from_image
from server.app import Sighting, decode_image
import cv2 as cv

voorbeeld = 5
path = 'C:/Users/Erwin2/OneDrive/Documenten/UA/Honours Program/Interdisciplinary Project/Salamanders/'

if __name__ == '__main__':

    if voorbeeld == 1:
        image1 = wrapped_imread(f'{path}2018/2018-Sal06.jpg')
        image2 = wrapped_imread(f'{path}2019/2019-Sal05.jpg')
        images = [image1, image2]
        for image in images:
            pose, succes = estimate_pose_from_image(image)
            print(pose)

            for body_part_name, (x, y, confidence) in pose.items():
                if confidence < 0.7:
                    continue
                cv.circle(image, (x, y), 5, (255, 255, 255), -1)
                if body_part_name != 'spine_middle':
                    cv.putText(image, body_part_name, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                               1)
                else:
                    cv.putText(image, body_part_name, (x, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                               1)

            cv.imshow('Pose Estimation', image)
            cv.waitKey(0)
        cv.destroyAllWindows()

    if voorbeeld == 2:
        image1 = wrapped_imread(f'{path}2018/2018-Sal06.jpg')
        image2 = wrapped_imread(f'{path}2019/2019-Sal05.jpg')
        images = [image1, image2]
        for image in images:
            pose, succes = estimate_pose_from_image(image)
            image_isolated = crop_image(image, coordinates_pose=pose, pose_estimation_evaluation=2)
            cv.imshow('Original', image)
            cv.imshow('Isolate belly', image_isolated)
            cv.waitKey(0)
        cv.destroyAllWindows()

    if voorbeeld == 3:
        image1 = wrapped_imread(f'{path}2018/2018-Sal20.jpg')
        image2 = wrapped_imread(f'{path}2019/2019-Sal05.jpg')
        images = [image2, image1]
        for image in images:
            image_isolated = crop_image(image, pose_estimation_evaluation=0)
            cv.imshow('Original', image)
            cv.imshow('Isolate belly', image_isolated)
            cv.waitKey(0)
        cv.destroyAllWindows()

    if voorbeeld == 4:
        image1 = wrapped_imread(f'{path}2019/2019-Sal18.jpg')
        image2 = wrapped_imread(f'{path}2019/2019-Sal05.jpg')
        images = [image1, image2]
        for image in images:
            pose, succes = estimate_pose_from_image(image)
            image_isolated = isolate_salamander(image, coordinates_pose=pose, pose_estimation_evaluation=2)
            dots = dot_detect_haar(image_isolated)

            drawn_image = draw_dots(image, dots)

            cv.imshow('Haar Cascade', drawn_image)
            cv.waitKey(0)
        cv.destroyAllWindows()

    if voorbeeld == 5:
        # Unknown image is the one that we just captured and photographed.
        unknown_image = wrapped_imread(f'{path}2018/2018-Sal04.jpg')

        coords_database = get_individuals_coords()

        list_coordinates, _, _ = image_to_canonical_representation(unknown_image)
        list_of_scores = compare_dot_patterns(list_coordinates, coords_database)

        # Convert the database in a dictionary for easy acces.
        database = {}
        for id, coords in coords_database:
            sighting: Sighting = get_dataclass(id, Sighting)
            image = get_file(sighting.image_id)
            # individual: Individual = get_dataclass(sighting.individual_id, Individual)
            database[id] = decode_image(image)
        # Plotting the results.
        display_results(unknown_image, database, list_of_scores)
