from typing import List, Tuple, Set

from src.utils import wrapped_imread
from scripts.Generating_extra_data import filenames_from_folder
from src.pattern_matching import compare_dot_patterns, display_results
from src.facade import image_to_canonical_representation
from server.database_interface import get_individuals_coords
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
        list_coordinates_image_from_database = image_to_canonical_representation(image_from_database)
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
    # for year in year_list:
    #     for sal in filenames_from_folder(
    #             f'{location}{year}'):
    #         img = wrapped_imread(f'{location}{year}{sal}')
    #
    #         database.append((img, sal))

    unknown_image = wrapped_imread(f'{location}2018/2018-Sal04.jpg')

    start_time = time()

    #other_database = get_db(database, 8, 0.1)
    coords_database = get_individuals_coords()

    list_coordinates = image_to_canonical_representation(unknown_image)
    list_of_scores = compare_dot_patterns(list_coordinates, coords_database)

    # Convert the database in a dictionary for easy acces.
    database = {name: image for image, name in database}
    # Plotting the results.
    display_results(unknown_image, database, list_of_scores)

    print(f"--- {time() - start_time} seconds ---")
