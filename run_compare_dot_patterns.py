import cv2

from dot_detection.dot_detect_haar import dot_detect_haar
from isolate_salamander import isolate_salamander
from Generating_extra_data import filenames_from_folder
from utils.heic_imread_wrapper import wrapped_imread
from Compare_dot_patterns import crop_image, compare_dot_patterns

database = []
year_list = ['2018/', '2019/', '2021/', '2022/', '2024/', 'Online/']
location = 'C:/Users/Erwin2/OneDrive/Documenten/UA/Honours Program/Interdisciplinary Project/Salamanders/'
if __name__ == '__main__':
    for year in year_list:
        for sal in filenames_from_folder(
                f'{location}{year}'):
            img = wrapped_imread(f'{location}{year}{sal}')

            database.append((img, sal))

    unknown_image = wrapped_imread(f'{location}2018/2018-Sal04.jpg')

    compare_dot_patterns(unknown_image, database)
