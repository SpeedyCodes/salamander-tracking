import gridfs
from bson import ObjectId
from pymongo import MongoClient, CursorType
from dataclasses import asdict, fields

from config import MONGO_CONNECTION_STRING
client = MongoClient(MONGO_CONNECTION_STRING)

db = client["Salamanders"]
individuals = db["individuals"]
sightings = db["sightings"]
bucket = gridfs.GridFSBucket(db)

def wrap_object_id(dataclass_instance):
    for field in fields(dataclass_instance):
        if "_id" in field.name:
            if hasattr(field.type, "_name") and field.type._name == "List":
                setattr(dataclass_instance, field.name, [ObjectId(id) for id in getattr(dataclass_instance, field.name)])
            elif getattr(dataclass_instance, field.name) is not None:
                setattr(dataclass_instance, field.name, ObjectId(getattr(dataclass_instance, field.name)))
    return dataclass_instance

def unwrap_object_id(dataclass_instance):
    for field in fields(dataclass_instance):
        if "_id" in field.name:
            if hasattr(field.type, "_name") and field.type._name == "List":
                setattr(dataclass_instance, field.name, [str(id) for id in getattr(dataclass_instance, field.name)])
            elif getattr(dataclass_instance, field.name) is not None:
                setattr(dataclass_instance, field.name, str(getattr(dataclass_instance, field.name)))
    return dataclass_instance

def get_individuals_coords():
    cursor: CursorType = sightings.find()
    # change the coords to a list of tuples to make it hashable
    list = [(str(doc["_id"]), [(coords[0], coords[1]) for coords in doc["coordinates"]]) for doc in cursor]
    return list

def get_individual_coords(name):
    doc = individuals.find_one({"name": name})
    return [(coords[0], coords[1]) for coords in doc["coordinates"]]
def store_dataclass(dataclass):
    dataclass = wrap_object_id(dataclass)
    dict = asdict(dataclass)
    dict.pop("_id")
    result = db[dataclass.collection_name].insert_one(dict)
    return str(result.inserted_id)

def get_dataclass(value, dataclass, field="_id"):
    if "_id" in field:
        value = ObjectId(value)
    doc = db[dataclass.collection_name].find_one({field: value})
    return unwrap_object_id(dataclass(**doc))

def set_field(_id, field, value, dataclass):
    if "_id" in field:
        value = ObjectId(value)
    db[dataclass.collection_name].update_one({"_id": ObjectId(_id)}, {"$set": {field: value}})

def get_all(dataclass):
    cursor: CursorType = db[dataclass.collection_name].find()
    list = [unwrap_object_id(dataclass(**doc)) for doc in cursor]
    return list

def store_file(file):
    _id = bucket.upload_from_stream("salamander_image", file)
    return str(_id)

def get_file(_id):
    file = bucket.open_download_stream(ObjectId(_id))
    return file.read()

