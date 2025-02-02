from flask import Flask, request, Response, render_template, jsonify
import cv2
import numpy as np

from server.models.image_pipeline import ImagePipeline
from server.models import Base
from server.models.named_location import NamedLocation
from src.facade import image_to_canonical_representation, match_canonical_representation_to_database
from server.database_interface import *
from dataclasses import dataclass, InitVar, asdict
from typing import List, Tuple, Set, Optional
from datetime import datetime
from server import db
from config import PG_CONNECTION_STRING
from flask_migrate import Migrate


app = Flask(__name__, static_folder='static', static_url_path='')
app.config['SQLALCHEMY_DATABASE_URI'] = PG_CONNECTION_STRING
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)


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
    image = decode_image(image_bytes)
    coordinates, quality = image_to_canonical_representation(image)
    coordinates, quality, intermediates = image_to_canonical_representation(image)


    candidates = match_canonical_representation_to_database(coordinates, 4)

    converted_list = [{
        "confidence": candidate[2],
        "sighting": get_sighting(int(candidate[4]))
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

        converted_list[i]["individual"] = get_individual(converted_list[i]["sighting"].individual_id)
        i += 1

    images = ImagePipeline(original_image=image_bytes, pose_estimation_image=encode_image(intermediates[0]), cropped_image=encode_image(intermediates[1]), dot_detection_image=encode_image(intermediates[2]), straightened_dots_image=encode_image(intermediates[3]))
    store_dataclass(images)
    sighting = Sighting(individual_id=None, image_id=images.id, coordinates=list(coordinates))
    sighting = store_dataclass(sighting)
    return {
        "sighting_id": sighting.id,
        "candidates": converted_list
    }


@app.route('/confirm/<string:sighting_id>', methods=['POST'])
def confirm(sighting_id):
    """
    Confirms the identity of the sighting with the given sighting_id,
    or creates a new individual if the salamander is not in the database.
    """
    individual_id = request.args.get('individual_id', type=int)
    location_id = request.args.get('location_id', type=int)
    body = request.json
    sighting = get_sighting(sighting_id)
    if not individual_id:  # if the individual is new
        individual = Individual(name=body["nickname"])
        individual = store_dataclass(individual)
    else:
        individual = db.session.get(Individual, individual_id)

    if location_id:
        sighting.location_id = location_id
    sighting.individual_id = individual.id
    sighting.date = datetime.strptime(body["spotted_at"], "%Y-%m-%dT%H:%M:%S.%f")
    db.session.commit()
    return Response(status=200)


@app.route('/individuals/<string:id>', methods=['GET'])
def info(id):
    """
    Returns the information of the salamander with the given id.
    """

    return jsonify(get_individual(id))


@app.route('/individuals', methods=['GET'])
def all_individuals():
    """
    Returns the information of all the salamanders in the database.
    """

    return get_individuals()


@app.route('/individuals/<string:id>/image', methods=['GET'])
def individual_image(id):
    """
    Returns an image of the salamander with the given id.
    """
    individual = db.session.get(Individual, id)
    sighting = db.session.query(Sighting).filter(Sighting.individual_id == individual.id).first()
    image = db.session.get(ImagePipeline, sighting.image_id).original_image
    return Response(image, mimetype='image/png')


@app.route('/sightings/<string:id>/image', methods=['GET'])
def sighting_image(id):
    """
    Returns the image of the sighting with the given id.
    """

    sighting: Sighting = get_sighting(id)
    image = db.session.get(ImagePipeline, sighting.image_id).original_image
    return Response(image, mimetype='image/png')

@app.route('/sightings/<string:id>/image/<string:intermediate_image_name>', methods=['GET'])
def intermediate_image(id, intermediate_image_name):
    """
    Returns the intermediate image of the sighting with the given id.
    """

    sighting: Sighting = get_sighting(id)
    pipeline = db.session.get(ImagePipeline, sighting.image_id)
    match intermediate_image_name:
        case "pose_estimation":
            image = pipeline.pose_estimation_image
        case "cropped":
            image = pipeline.cropped_image
        case "dot_detection":
            image = pipeline.dot_detection_image
        case "straightened_dots":
            image = pipeline.straightened_dots_image
        case _:
            return Response(status=404)
    return Response(image, mimetype='image/png')

@app.route('/sightings', methods=['GET'])
def get_sightings():
    """
    Returns the information of all the sightings in the database.
    """

    query = db.session.query(Sighting).filter(Sighting.individual_id != None)

    if "individual_id" in request.args:
        query = query.filter(Sighting.individual_id == request.args["individual_id"])

    return query.all()

@app.route('/sightings/<string:id>', methods=['DELETE'])
def delete_sighting(id):
    """
    Deletes the sighting with the given id.
    """

    sighting = get_sighting(id)
    db.session.delete(sighting)
    db.session.commit()
    return Response(status=200)

@app.route('/locations', methods=['POST'])
def add_location():
    """
    Adds a location to the database.
    """
    body = request.json
    location = NamedLocation(name=body["name"], precise_location=body["precise_location"])
    location = store_dataclass(location)
    return asdict(location)

@app.route('/locations', methods=['GET'])
def get_locations():
    """
    Returns all the locations in the database.
    """
    return db.session.query(NamedLocation).all()

@app.after_request
def after_request(response):
    # cors stuff to make flutter-web work
    response.headers['Access-Control-Allow-Origin'] = request.origin
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE'
    response.headers['Vary'] = 'Origin'
    return response

@app.route("/")
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
