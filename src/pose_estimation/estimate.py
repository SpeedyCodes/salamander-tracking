from typing import Dict, Tuple
import tempfile
import numpy as np
import cv2
import os
import psutil
from subprocess import Popen, PIPE
from uuid import uuid4
from sys import executable
from time import sleep
from src.pose_estimation.common_paths import working_dir, csv_file_path, h5_file_path, pickle_file_path, patience

# This file abstracts away the process of analyzing an image using DeepLabCut.
# Lots of filesystem-related mess
# because DeepLabCut can neither read from an in-memory image nor return the results from the function itself
# also, we run it in a separate process to survive crashes from DeepLabCut



process: Popen = None
psProcess: psutil.Process = None


def restart_subprocess():
    global process
    global psProcess
    process = Popen([executable, "src/pose_estimation/deeplabcut_runner.py"],
                    stdout=PIPE, stderr=PIPE, text=True)

    psProcess = psutil.Process(pid=process.pid)

def estimate_pose_from_image(image: np.ndarray) -> Tuple[Dict[str, Tuple[int, int, float]], bool]:
    """
    Estimate the pose of a salamander in an image using DeepLabCut.
    :param image: The image to analyze
    :return: A dictionary mapping body part names to their estimated positions and confidences:
     {body_part_name: (x, y, confidence)} and a boolean indicating whether the analysis was successful
    """
    global process
    global psProcess

    # delete all files in the working directory
    for path in os.listdir(working_dir):
        os.remove(f"{working_dir}/{path}")


    image_name = f"{working_dir}/image{uuid4()}.jpg"
    # save the image to the working directory so DLC can read it
    cv2.imwrite(image_name, image)
    # make sure the DLC process is running
    if process is None or process.poll() is not None:
        restart_subprocess()
    else:
        psProcess.resume()

    # wait for the CSV file to be created or the process to crash
    poll = None
    while poll is None:
        poll = process.poll()
        # check if file exists
        if os.path.exists(csv_file_path):
            file_being_used = True
            while file_being_used:
                try:
                    os.rename(csv_file_path, csv_file_path)
                    file_being_used = False
                except OSError:  # file is in use
                    file_being_used = True
                    sleep(patience)
            psProcess.suspend()  # suspend the process, we don't need it for now
            break
        sleep(patience)

    if poll != 0 and poll is not None:  # error code: process crashed
        restart_subprocess()
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
    # wait for every file to become available to make sure it is deleted
    # the cleanup at the beginning of the function sometimes misses files because they are not yet available
    for path in [image_name, csv_file_path, h5_file_path, pickle_file_path]:
        while not os.path.exists(path):
            sleep(patience)
            pass
        while True:
            try:
                os.remove(path)
                break
            except OSError:  # file is in use
                sleep(patience)
                pass

    return body_parts, True
