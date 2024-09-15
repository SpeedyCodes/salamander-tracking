# assembles the training data for the third version of the DLC model


import os
import shutil
from src.utils import wrapped_imread
import cv2
import deeplabcut as dlc
import csv
from random import randint


source_dir = 'training/dlc/salamander-jesse-2024-08-19/labeled-data/m3v1mp4/'
dest_dir = 'training/dlc/salamander-jesse-2024-09-15/labeled-data/m3v1mp4/'
old_csv_file = 'training/dlc/salamander-jesse-2024-08-19/labeled-data/m3v1mp4/CollectedData_jesse.csv'
#open the csv file for editing
with open(old_csv_file, 'r') as file:
    reader = csv.reader(file)
    csv_data = list(reader)

# clean the destination directory
for file in os.listdir(dest_dir):
    os.remove(os.path.join(dest_dir, file))

# add all the old training data and some new
filestocopy = [os.path.join(source_dir, file) for file in os.listdir(source_dir)]
filestocopy += [os.path.join("input/2020_new/", file) for file in os.listdir("input/2020_new/")][:-1]
filestocopy += [os.path.join("input/2021/", file) for file in os.listdir("input/2021/")]
filestocopy += [os.path.join("input/2021_new/", file) for file in os.listdir("input/2021_new/")][:-1]

for file in filestocopy:
    # we need new versions of these
    if file.endswith('.csv') or file.endswith('.h5') or "2022-Sal24.jpg" in file:
        continue

    img = wrapped_imread(file)
    # make sure the image is detected by DLC
    new_filename = file.replace('.HEIC', '.jpg')
    new_filename = new_filename.replace('.jpeg', '.jpg')

    # crop some large images
    match new_filename:
        case "IMG_3004.jpg":
            img = img[2000:img.shape[0]-200, 0:img.shape[1]-1500]
        case "IMG_3005.jpg":
            img = img[500:img.shape[0]-1000, 500:img.shape[1]-1000]
        case "IMG_3006.jpg":
            img = img[1400:img.shape[0]-400, 500:img.shape[1]-1000]
        case "IMG_3007.jpg":
            img = img[1000:img.shape[0]-1000, 0:img.shape[1]-1500]
        case "IMG_3008.jpg":
            img = img[1000:img.shape[0]-500, 0:img.shape[1]-1500]
        case "IMG_3021.jpg":
            img = img[1500:img.shape[0], 0:img.shape[1]-1300]
        case "IMG_3022.jpg":
            img = img[1300:img.shape[0]-500, 0:img.shape[1]-1500]
        case "IMG_3023.jpg":
            img = img[1100:img.shape[0]-700, 0:img.shape[1]-1500]
        case "IMG_3024.jpg":
            img = img[1700:img.shape[0], 0:img.shape[1]-1500]
        case "IMG_3025.jpg":
            img = img[1300:img.shape[0]-500, 0:img.shape[1]-1500]
        case _:
            pass

    # rotate the images randomly to prevent the model from relying on a specific orientation

    image_height, image_width = img.shape[:2]
    for j in range(randint(0, 4)):  # rotate between 0 and 3 times 90 degrees
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # rotate the existing labels too - I don't want to do those again
        for row in csv_data[3:]:
            if row[2] in file:
                for i in range(3, len(row), 2):
                    if row[i] == '':
                        continue
                    old_x = float(row[i])
                    old_y = float(row[i+1])
                    new_x = image_height - old_y
                    new_y = old_x
                    row[i] = str(new_x)
                    row[i+1] = str(new_y)
        image_height, image_width = image_width, image_height

    cv2.imwrite(os.path.join(dest_dir, new_filename.split("/")[-1]), img)

# write the new csv file with rotated labels
new_csv_file = os.path.join(dest_dir, 'CollectedData_jesse.csv')
with open(new_csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)
dlc.convertcsv2h5('D:/School/UAntwerpen/Honours/Salamander/salamander-tracking/training/dlc/salamander-jesse-2024-09-15/config.yaml', scorer= 'jesse')
        
print("amount of files in destination directory:", len(os.listdir(dest_dir)))
