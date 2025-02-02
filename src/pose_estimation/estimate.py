from typing import Dict, Tuple
import numpy as np
import cv2
import os
from uuid import uuid4
from src.pose_estimation.common_paths import working_dir, csv_file_path, h5_file_path, pickle_file_path, patience, \
    config_file_path
from huey import RedisHuey
import deeplabcut as dlc
from config import huey_immediate, pose_estimation_timeout

# This file abstracts away the process of analyzing an image using DeepLabCut.
# Lots of filesystem-related mess
# because DeepLabCut can neither read from an in-memory image nor return the results from the function itself
# also, we run it in a separate Huey task to survive crashes from DeepLabCut

huey = RedisHuey("redis-server", host="localhost")
huey.immediate = huey_immediate

@huey.task()
def _estimate(dir):
    dlc.analyze_time_lapse_frames(config_file_path, dir, save_as_csv=True, frametype='jpg')
    return True

def cleanup_dir(dir):
    for file in os.listdir(dir):
        os.remove(f"{dir}/{file}")
    os.rmdir(dir)


def estimate_pose_from_image(image: np.ndarray) -> Tuple[Dict[str, Tuple[int, int, float]], bool]:
    """
    Estimate the pose of a salamander in an image using DeepLabCut.
    :param image: The image to analyze
    :return: A dictionary mapping body part names to their estimated positions and confidences:
     {body_part_name: (x, y, confidence)} and a boolean indicating whether the analysis was successful
    """

    uuid = uuid4()
    image_dir = f"{working_dir}/{uuid}"
    os.makedirs(image_dir)
    image_path = f"{image_dir}/image.jpg"
    # save the image to the working directory so DLC can read it
    cv2.imwrite(image_path, image)
    task = _estimate(image_dir)
    try:
        result = task(blocking=True, timeout=pose_estimation_timeout)
    except Exception as e:
        cleanup_dir(image_dir)
        return {}, False

    # read the CSV file that DLC generated
    csv_file_path = f"{image_dir}/{uuid}DLC_resnet50_salamanderAug19shuffle1_300000.csv"
    lines = []
    with open(csv_file_path, "r") as file:
        for line in file:
            parts = line.split(",")
            lines.append(parts)

    cleanup_dir(image_dir)

    body_parts = {}
    for i in range(1, len(lines[1]), 3):
        body_part_name = lines[1][i]
        x, y, confidence = lines[3][i:i + 3]
        body_parts[body_part_name] = (int(float(x)), int(float(y)), float(confidence))

    return body_parts, True

def draw_pose(image, pose):
    new_image = image.copy()
    for body_part_name, (x, y, confidence) in pose.items():
        if confidence < 0.6:
            continue
        cv2.circle(new_image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(new_image, body_part_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return new_image
