from scripts.Generating_extra_data import filenames_from_folder
from src.utils import wrapped_imread
from src.pattern_matching import compare_dot_patterns
from time import time

database = []
year_list = ['2018/', '2019/', '2021/', '2022/', '2024/']
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
