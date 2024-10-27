from sqlalchemy import select, insert

from server.models.individual import Individual
from server.models.sighting import Sighting
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

def get_sighting(sighting_id: int) -> Sighting:
    return db.session.get(Sighting, sighting_id)

def get_individual(individual_id: int) -> Individual:
    return db.session.get(Individual, individual_id)

def get_individuals():
    return db.session.query(Individual).all()