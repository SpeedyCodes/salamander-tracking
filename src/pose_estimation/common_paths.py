import tempfile
import os
# make new a new directory to save the image
tempdir = tempfile.gettempdir()
working_dir = f"{tempdir}/salamander-tracking"
os.makedirs(working_dir, exist_ok=True)
# path to the files that DLC will generate
output_file = f"{working_dir}/salamander-trackingDLC_resnet50_salamanderAug19shuffle1_210000"
csv_file_path = f"{output_file}.csv"
h5_file_path = f"{output_file}.h5"
pickle_file_path = f"{output_file}_meta.pickle"
# path to DLC project config file
config_file_path = "training/dlc/salamander-jesse-2024-09-15/config.yaml"
# amount of time to wait inbetween polling for files
patience = 0.01
