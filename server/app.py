from flask import Flask, request, Response
import cv2
import numpy as np
from src.facade import image_to_canonical_representation, match_canonical_representation_to_database
from server.database_interface import *
from dataclasses import dataclass, InitVar, asdict
from typing import List, Tuple, Set, Optional
from datetime import datetime

app = Flask(__name__)


@dataclass
class Individual:
    name: str
    sighting_ids: List[str]
    _id: Optional[str] = None
    coordinates: Optional[List[tuple[float, float]]] = None
    collection_name: InitVar[str] = "individuals"


@dataclass
class Sighting:
    individual_id: Optional[str]
    image_id: str
    _id: Optional[str] = None
    coordinates: Optional[List[tuple[float, float]]] = None
    date: Optional[datetime] = None
    collection_name: InitVar[str] = "sightings"


def encode_image(image: np.ndarray):
    return cv2.imencode('.jpg', image)[1].tobytes()


def decode_image(bytestring: bytes):
    image = np.frombuffer(bytestring, dtype=np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)


@app.route('/store_sighting', methods=['POST'])
def recognize():
    """
    Performs recognition and returns candidates with their confidence levels.
    The submitted image is saved as a sighting to be confirmed later.
    """
    image_bytes = request.data
    image_id = store_file(image_bytes)
    image = decode_image(image_bytes)
    coordinates, quality = image_to_canonical_representation(image)


    candidates = match_canonical_representation_to_database(coordinates, 4)

    converted_list = [{
        "confidence": candidate[2],
        "sighting": get_dataclass(candidate[4], Sighting)
    }
        for candidate in candidates]
    i = 0
    while i < len(converted_list):
        sighting = converted_list[i]["sighting"]
        if sighting.individual_id is None:
            converted_list.pop(i)
            continue
        # filter out all but the first candidate of every individual_id
        # also, remove candidates that have no individual_id
        j = i + 1
        while j < len(converted_list):
            if sighting.individual_id == converted_list[j]["sighting"].individual_id or converted_list[j]["sighting"].individual_id is None:
                converted_list.pop(j)
            else:
                j += 1

        converted_list[i]["individual"] = get_dataclass(converted_list[i]["sighting"].individual_id, Individual)
        i += 1


    sighting = Sighting(individual_id=None, image_id=image_id, coordinates=list(coordinates))
    sighting_id = store_dataclass(sighting)
    return {
        "sighting_id": sighting_id,
        "candidates": converted_list
    }


@app.route('/confirm/<string:sighting_id>', methods=['POST'])
def confirm(sighting_id):
    """
    Confirms the identity of the sighting with the given sighting_id,
    or creates a new individual if the salamander is not in the database.
    """
    individual_id = request.args.get('individual_id')
    body = request.json
    if not individual_id:  # if the individual is new
        sighting: Sighting = get_dataclass(sighting_id, Sighting)
        individual = Individual(name=body["nickname"], coordinates=sighting.coordinates, sighting_ids=[sighting._id])
        individual_id = store_dataclass(individual)
    else:  # if the individual is already in the database
        db["individuals"].update_one({"_id": ObjectId(individual_id)},
                                     {"$push": {"sighting_ids": ObjectId(sighting_id)}})
    set_field(sighting_id, "individual_id", individual_id, Sighting)
    set_field(sighting_id, "date", body["spotted_at"], Sighting)
    return Response(status=200)


@app.route('/individuals/<string:id>', methods=['GET'])
def info(id):
    """
    Returns the information of the salamander with the given id.
    """

    individual = get_dataclass(id, Individual)
    return asdict(individual)


@app.route('/individuals', methods=['GET'])
def all_individuals():
    """
    Returns the information of all the salamanders in the database.
    """

    individuals = get_all(Individual)
    return [asdict(individual) for individual in individuals]


@app.route('/individuals/<string:id>/image', methods=['GET'])
def individual_image(id):
    """
    Returns an image of the salamander with the given id.
    """

    # get one of the sightings where the individual_id is the given id
    cursor: CursorType = sightings.find({"individual_id": ObjectId(id)})
    list = [Sighting(**doc) for doc in cursor]
    sighting: Sighting = list[0]
    image = get_file(sighting.image_id)
    return Response(image, mimetype='image/png')


@app.route('/sightings/<string:id>/image', methods=['GET'])
def sighting_image(id):
    """
    Returns the image of the sighting with the given id.
    """

    sighting: Sighting = get_dataclass(id, Sighting)
    image = get_file(sighting.image_id)
    return Response(image, mimetype='image/png')

@app.route('/sightings', methods=['GET'])
def get_sightings():
    """
    Returns the information of all the sightings in the database.
    """

    sightings = get_all(Sighting)
    list = [asdict(sighting) for sighting in sightings if sighting.individual_id is not None]

    if "individual_id" in request.args:
        list = [sighting for sighting in list if sighting["individual_id"] == request.args["individual_id"]]
    for sighting in list:
        sighting.pop("coordinates")
        sighting["individual_name"] = get_dataclass(sighting["individual_id"], Individual).name

    return list



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
