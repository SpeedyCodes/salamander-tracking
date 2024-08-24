from typing import Tuple, List
import numpy as np

from src.utils import wrapped_imread
def read_annotation_file(file: str) -> Tuple[List[np.ndarray], List[List[Tuple[int, int, int, int]]]]:
    images = []
    rectangles = []

    with open(file, 'r') as file:
        for line in file:
            parts = line.split(' ')
            image = wrapped_imread(parts[0])
            images.append(image)
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
    return images, rectangles