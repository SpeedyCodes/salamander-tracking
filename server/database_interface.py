from sqlalchemy import select, insert

from models.individual import Individual
from models.sighting import Sighting
from server import db


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
    individuals = db.session.scalars(select(Individual)).all()
    # change the coords to a list of tuples to make it hashable
    return [(str(individual["_id"]), [(coords[0], coords[1]) for coords in individual["coordinates"]]) for individual in individuals]

def store_dataclass(object):
    db.session.add(object)
    db.session.commit()
    return object

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

