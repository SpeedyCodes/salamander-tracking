from typing import List, Tuple, Set

from scripts.Generating_extra_data import filenames_from_folder
from src.dot_detection import dot_detect_haar
from src.preprocessing import normalise_coordinates, crop_image
from src.utils import wrapped_imread
from src.pattern_matching import compare_dot_patterns, display_results
from src.pattern_matching.Compare_dot_patterns import select_points_to_be_matched
from time import time
import threading

database = []
year_list = ['2018/', '2019/', '2021/', '2022/', '2024/']
location = 'C:/Users/Erwin2/OneDrive/Documenten/UA/Honours Program/Interdisciplinary Project/Salamanders/'


def get_db(database, thread_count, tol) -> List[Tuple[Set[Tuple[float, float]], str]]:
    start_time = time()
    threads = []

    database_of_coordinates = []
    def image_to_coords(image_from_database, name_image_from_database):
        image_from_database = crop_image(image_from_database)
        list_haar_cascade = dot_detect_haar(image_from_database)
        list_coordinates_image_from_database = normalise_coordinates(list_haar_cascade, image_from_database.shape)
        list_coordinates_image_from_database = select_points_to_be_matched(list_coordinates_image_from_database,
                                                                           tol=3 * tol)
        database_of_coordinates.append((list_coordinates_image_from_database, name_image_from_database))

    if thread_count == 1:  # no threading
        for image_database, name_image_database in database:
            image_to_coords(image_database, name_image_database)
    else:  # threaded execution
        def thread_function(start, end):
            for j in range(start, end):
                image_to_coords(database[j][0], database[j][1])

        for i in range(thread_count):
            threads.append(threading.Thread(target=thread_function, args=(
                i * len(database) // thread_count, (i + 1) * len(database) // thread_count)))
            threads[-1].start()

        for thread in threads:
            thread.join()
    print(f"Time for loading DB: {time() - start_time} seconds")

    return database_of_coordinates


if __name__ == '__main__':
    for year in year_list:
        for sal in filenames_from_folder(
                f'{location}{year}'):
            img = wrapped_imread(f'{location}{year}{sal}')

            database.append((img, sal))

    unknown_image = wrapped_imread(f'{location}2018/2018-Sal04.jpg')

    start_time = time()

    # First, crop the image to the essential part.
    image_crop = crop_image(unknown_image)
    list_haar_cascade = dot_detect_haar(image_crop)
    list_coordinates = normalise_coordinates(list_haar_cascade, image_crop.shape)
    list_of_scores = compare_dot_patterns(list_coordinates, get_db(database, 8, 0.01))

    # Convert the database in a dictionary for easy acces.
    database = {name: image for image, name in database}
    # Plotting the results.
    display_results(unknown_image, database, list_of_scores)

    print(f"--- {time() - start_time} seconds ---")
