from Generating_extra_data import filenames_from_folder
from utils.heic_imread_wrapper import wrapped_imread
from Compare_dot_patterns import compare_dot_patterns
from time import time

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

    start_time = time()
    compare_dot_patterns(unknown_image, database)

    print(f"--- {time() - start_time} seconds ---")
