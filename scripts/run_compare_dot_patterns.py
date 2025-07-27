from typing import List, Tuple, Set

from src.utils import wrapped_imread
from scripts.Generating_extra_data import filenames_from_folder
from src.pattern_matching import compare_dot_patterns, display_results
from src.facade import image_to_canonical_representation
from server.database_interface import get_individuals_coords
from time import time
import threading
database = []
year_list = ['2025/2025-03-08 spoor OK/']
location = 'C:/backup1/rune/UA/Honours Program/Interdisciplinary Project/Salamanders/'



def get_db(database, thread_count, tol) -> List[Tuple[Set[Tuple[float, float]], str]]:
    start_time = time()
    threads = []

    database_of_coordinates = []
    def image_to_coords(image_from_database, name_image_from_database):
        list_coordinates_image_from_database = image_to_canonical_representation(image_from_database)
        database_of_coordinates.append((name_image_from_database, list_coordinates_image_from_database))

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

            if 'IMG' in sal:
                continue
            else:
                database.append((img, sal))

    unknown_image = wrapped_imread(f'{location}2025/2025-03-08 spoor OK/sal 25-03-08 spoor 1 tris small.jpg')

    start_time = time()

    coords_database = get_db(database, 1, 0.1)
    coords_db = []
    for name, rest in coords_database:
        coords_db.append((name, rest[0]))
    # coords_database = get_individuals_coords()

    list_coordinates, _, _ = image_to_canonical_representation(unknown_image)
    list_of_scores = compare_dot_patterns(list_coordinates, coords_db)

    # Convert the database in a dictionary for easy access.
    database = {name: image for image, name in database}
    # Plotting the results.
    display_results(unknown_image, database, list_of_scores)

    print(f"--- {time() - start_time} seconds ---")
