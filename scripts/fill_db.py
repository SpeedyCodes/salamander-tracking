from Generating_extra_data import filenames_from_folder
from run_compare_dot_patterns import get_db
from src.utils import wrapped_imread

# populates the db with all the salamanders we have
from pymongo import MongoClient

from server.config import MONGO_CONNECTION_STRING
client = MongoClient(MONGO_CONNECTION_STRING)

db = client["Salamanders"]
collection = db["individuals"]


images = []
location = "D:/School/UAntwerpen/Honours/Salamander/salamander-tracking/input/"
for year in ["2018/", "2019/", "2021/", "2022/", "2024/"]:
    for sal in filenames_from_folder(f'{location}{year}'):
        img = wrapped_imread(f'{location}{year}{sal}')

        images.append((img, sal))

coords_and_name = get_db(images, 8, 0.01)
collection.insert_many([{"name": name, "coords": list(coords)} for coords, name in coords_and_name])
