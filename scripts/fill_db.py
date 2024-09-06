from Generating_extra_data import filenames_from_folder
from run_compare_dot_patterns import get_db
from src.utils import wrapped_imread
from server.database_interface import *
from server.app import Individual, Sighting, encode_image
from datetime import datetime

# populates the db with all the salamanders we have



images = []
location = "C:/Users/Erwin2/OneDrive/Documenten/UA/Honours Program/Interdisciplinary Project/Salamanders/"
for year in ["2018/", "2019/", "2021/", "2022/", "2024/"]:
    for sal in filenames_from_folder(f'{location}{year}'):
        img = wrapped_imread(f'{location}{year}{sal}')

        images.append((img, sal))

coords_and_name = get_db(images, 1, 0.01)

index = 0

for coords, name in coords_and_name:
    image_id = store_file(encode_image(images[index][0]))
    sighting = Sighting(individual_id=None, image_id=image_id)
    sighting_id = store_dataclass(sighting)

    individual = Individual(name=name, coordinates=sighting.coordinates, sighting_ids=[sighting._id])
    individual_id = store_dataclass(individual)

    set_field(sighting_id, "individual_id", individual_id, Sighting)
    set_field(sighting_id, "date", datetime.now(), Sighting)
    print(f"Stored {name}")
    index += 1