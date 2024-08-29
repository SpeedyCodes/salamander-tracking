import os
from time import sleep
import deeplabcut as dlc
from common_paths import working_dir, config_file_path

# this file is run as a separate process to allow the main process to survive crashes from DLC


lastfilename = ''
while True:
    # as long as there is not exactly one new jpg file in the working directory, wait
    # (this isn't actually polling constantly, the process gets suspended when not needed)
    contents = os.listdir(working_dir)
    while len(contents) != 1 or contents[0] == lastfilename or not contents[0].endswith('.jpg'):
        sleep(0.1)
        contents = os.listdir(working_dir)
    lastfilename = contents[0]

    dlc.analyze_time_lapse_frames(config_file_path, working_dir, save_as_csv=True, frametype='jpg')

