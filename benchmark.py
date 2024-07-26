from typing import Tuple, List
import numpy as np
from Convert_image_to_stips import convert_image_to_coordinate_stips
from dot_detection.dot_detect_haar import dot_detect_haar
from math import sqrt

def threshold_adapter(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    :param image: the image to detect the dots in
    :return: the detected dots in the image: (x, y, width, height)
    """
    results = convert_image_to_coordinate_stips(image)

    output = []

    for x, y, radius, good in results:
        if good:
            output.append((x - radius, y - radius, 2 * radius, 2 * radius))

    return output


def iou(rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]) -> float:
    """
    :param rect1: the first rectangle
    :param rect2: the second rectangle
    :return: the intersection over union of the two rectangles
    """
    x1, y1, width1, height1 = rect1
    x2, y2, width2, height2 = rect2

    x_intersection = max(0, min(x1 + width1, x2 + width2) - max(x1, x2))
    y_intersection = max(0, min(y1 + height1, y2 + height2) - max(y1, y2))

    intersection = x_intersection * y_intersection
    union = width1 * height1 + width2 * height2 - intersection

    return intersection / union

def rect_center_distance_measure(rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]) -> float:
    """
    :param rect1: the first rectangle
    :param rect2: the second rectangle
    :return: a value between 0 and 1, where 1 means the rectangles are very close and 0 means they are very far apart
    """
    x1, y1, width1, height1 = rect1
    x2, y2, width2, height2 = rect2

    center1 = (x1 + width1 / 2, y1 + height1 / 2)
    center2 = (x2 + width2 / 2, y2 + height2 / 2)

    distance = sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    average_diagonal = (sqrt(width1 ** 2 + height1 ** 2) + sqrt(width2 ** 2 + height2 ** 2)) / 2

    return max(0, (average_diagonal - distance)/average_diagonal)


def dot_detect_benchmark(measure, func, images, ground_truth) -> float:
    """
    :param measure: the function to measure the similarity of two rectangles
    :param func: the function to benchmark, must take in an image and return the detected dots
    :param images: a list of images to test the function on
    :param ground_truth: a list of, for every image, a list of rectangles that contain the dots
    :return: the accuracy of the function on the images (0-1)
    """
    diff_sum = 0
    for index, image in enumerate(images):
        detected_dots = func(image)  # the detected dots
        real_rectangles = ground_truth[index]  # the dots marked manually

        if len(real_rectangles) == 0:
            diff_sum += 1
            continue

        # find the best matches

        matches = []

        for detected_index, detected_dot in enumerate(detected_dots):
            for real_index, real_rectangle in enumerate(real_rectangles):
                diff = measure(detected_dot, real_rectangle)
                matches.append((diff, detected_index, real_index))

        matches.sort(key=lambda x: x[0], reverse=True)  # sort the matches by diff in descending order

        detected_occupied = set()
        real_occupied = set()
        real_diffs = [0] * len(real_rectangles)

        while len(matches) > 0:
            diff, detected_index, real_index = matches.pop(0)  # take the best match

            # if the detected dot and the real rectangle are not already matched
            if detected_index not in detected_occupied and real_index not in real_occupied:
                # mark them as matched
                detected_occupied.add(detected_index)
                real_occupied.add(real_index)
                real_diffs[real_index] = diff  # store the diff of the match

                if len(detected_occupied) == len(detected_dots) or len(real_occupied) == len(real_rectangles):
                    break

        diff_sum += sum(real_diffs) / len(real_diffs)

    return diff_sum / len(images)


if __name__ == '__main__':
    from utils.heic_imread_wrapper import wrapped_imread

    images = []
    rectangles = []

    # load the rectangles and images from annotation file with the following format:
    # image_path amount_of_rectangles x1 y1 width1 height1 x2 y2 width2 height ...
    #with open('benchmark_anno.txt', 'r') as file:
    with open('training/haar_cascade/merged_annotations.txt', 'r') as file:
        for line in file:
            parts = line.split(' ')
            image = wrapped_imread(parts[0])
            images.append(image)

            i = 0
            count = parts[1]
            image_rectangles = []

            parts = [int(part) for part in parts[2:]]  # skip image name and count, convert to int

            for i in range(int(count)):
                x = parts[i * 4 + 0]
                y = parts[i * 4 + 1]
                width = parts[i * 4 + 2]
                height = parts[i * 4 + 3]
                image_rectangles.append((x, y, width, height))

            rectangles.append(image_rectangles)

    # calculate the accuracy of the haar cascade
    accuracy = dot_detect_benchmark(iou, dot_detect_haar, images, rectangles)
    print(f"Accuracy with haar cascade according to iou: {accuracy}")
    accuracy = dot_detect_benchmark(iou, threshold_adapter, images, rectangles)
    print(f"Accuracy with threshold according to iou: {accuracy}")
    accuracy = dot_detect_benchmark(rect_center_distance_measure, dot_detect_haar, images, rectangles)
    print(f"Accuracy with haar cascade according to rect_center_distance_measure: {accuracy}")
    accuracy = dot_detect_benchmark(rect_center_distance_measure, threshold_adapter, images, rectangles)
    print(f"Accuracy with threshold according to rect_center_distance_measure: {accuracy}")
    
