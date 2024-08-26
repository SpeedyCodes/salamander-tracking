from typing import Dict, Tuple
import tempfile
import numpy as np
import cv2
import os
import subprocess
# This file abstracts away the process of analyzing an image using DeepLabCut.
# Lots of filesystem-related mess
# because DeepLabCut can neither read from an in-memory image nor return the results from the function itself

# make new a new directory to save the image
tempdir = tempfile.gettempdir()
working_dir = f"{tempdir}/salamander-tracking"
os.makedirs(working_dir, exist_ok=True)
# path to DLC project config file
config_file_path = "training/dlc/salamander-jesse-2024-08-19/config.yaml"
# path to the files that DLC will generate
output_file = f"{working_dir}/salamander-trackingDLC_resnet50_salamanderAug19shuffle1_210000"
csv_file_path = f"{output_file}.csv"
h5_file_path = f"{output_file}.h5"
pickle_file_path = f"{output_file}_meta.pickle"


def estimate_pose_from_image(image: np.ndarray) -> Tuple[Dict[str, Tuple[int, int, float]], bool]:
    """
    Estimate the pose of a salamander in an image using DeepLabCut.
    :param image: The image to analyze
    :return: A dictionary mapping body part names to their estimated positions and confidences:
     {body_part_name: (x, y, confidence)} and a boolean indicating whether the analysis was successful
    """

    # save the image to the working directory so DLC can read it
    cv2.imwrite(f"{working_dir}/image.jpg", image)
    # analyze the image using DLC, in separate subprocess to survive crashes
    p = subprocess.Popen(["venv/Scripts/python.exe",  "src/pose_estimation/deeplabcut_runner.py"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.wait() != 0: # error code
        return {}, False

    # read the CSV file that DLC generated
    lines = []
    with open(csv_file_path, "r") as file:
        for line in file:
            parts = line.split(",")
            lines.append(parts)

    body_parts = {}
    for i in range(1, len(lines[1]), 3):
        body_part_name = lines[1][i]
        x, y, confidence = lines[3][i:i + 3]
        body_parts[body_part_name] = (int(float(x)), int(float(y)), float(confidence))

    # clean up the working directory
    for path in [f"{working_dir}/image.jpg", csv_file_path, h5_file_path, pickle_file_path]:
        if os.path.exists(path):
            os.remove(path)

    return body_parts, True
