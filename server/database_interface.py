from dataclasses import asdict
from typing import Optional

from sqlalchemy import select, insert
from server.models.individual import Individual
from server.models.sighting import Sighting
from server import db


def get_individuals_coords(location_id: int | None = None):
    query = db.session.query(Sighting)
    if location_id is not None:
        query = query.filter(Sighting.location_id == location_id)
    sightings = query.all()
    # change the coords to a list of tuples to make it hashable
    return [(str(sighting.id), [(coords[0], coords[1]) for coords in sighting.coordinates]) for sighting in sightings]

def store_dataclass(object):
    db.session.add(object)
    db.session.commit()
    return object

def get_sighting(sighting_id: int) -> Sighting:
    return db.session.get(Sighting, sighting_id)

def get_individual(individual_id: int) -> Individual:
    return db.session.get(Individual, individual_id)

def get_individuals(location_id: Optional[int] = None) -> list[Individual]:
    query = db.session.query(Individual)
    if location_id is not None:
        # get all individuals for which a sighting exists at the location id
        query = (
            query
            .join(Sighting)
            .filter(Sighting.location_id == location_id)
            .distinct()
        )
    return query.all()


def get_sightings(individual_id: Optional[int] = None, location_id: Optional[int] = None) -> list[Sighting]:
    query = db.session.query(Sighting).filter(Sighting.individual_id != None)
    if individual_id is not None:
        query = query.filter(Sighting.individual_id == individual_id)
    if location_id is not None:
        query = query.filter(Sighting.location_id == location_id)
    return query.all()

def confirm_sighting(sighting_id, date, individual_id):
    individual = db.session.get(Individual, individual_id)
    if sighting_id not in individual.sighting_ids:
        individual.sighting_ids.append(sighting_id)
    sighting = db.session.get(Sighting, sighting_id)
    sighting.individual_id = individual_id
    sighting.date = date

def sighting_asdict(sighting: Sighting) -> dict:
    if sighting.date is not None:
        sighting.date = sighting.date.strftime("%Y-%m-%dT%H:%M:%S.%f")
    return asdict(sighting)
