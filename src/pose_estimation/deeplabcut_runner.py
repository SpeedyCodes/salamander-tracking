import tempfile
import os
import deeplabcut as dlc

# this file is run as a separate process to allow the main process to survive crashes

# make new a new directory to save the image
tempdir = tempfile.gettempdir()
working_dir = f"{tempdir}/salamander-tracking"
os.makedirs(working_dir, exist_ok=True)
# path to DLC project config file
config_file_path = "training/dlc/salamander-jesse-2024-08-19/config.yaml"

dlc.analyze_time_lapse_frames(config_file_path, working_dir, save_as_csv=True, frametype='jpg')
