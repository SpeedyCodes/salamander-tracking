from flask import Flask, request, Response, render_template, jsonify
import cv2
import numpy as np

from server.models.image_pipeline import ImagePipeline
from server.models import Base
from server.models.named_location import NamedLocation
from server.models.password import Password
from src.facade import image_to_canonical_representation, match_canonical_representation_to_database
from server.database_interface import *
from dataclasses import dataclass, InitVar, asdict
from typing import List, Tuple, Set, Optional
from datetime import datetime, timedelta
from server import db
from config import PG_CONNECTION_STRING, jwt_secret
from flask_migrate import Migrate
import bcrypt
import jwt

from src.utils import ImageQuality

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

    location_id = request.args.get('location_id', type=int)
    image_bytes = request.data
    image = decode_image(image_bytes)
    coordinates, quality, intermediates = image_to_canonical_representation(image)
    if quality == ImageQuality.BAD: # if the image is too bad to process, return a 400 and don't store the image
        return Response(status=400)


    candidates = match_canonical_representation_to_database(coordinates, 4, location_id)

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

    pose_estimation_image = encode_image(intermediates[0]) if intermediates[0] is not None else None
    cropped_image = encode_image(intermediates[1]) if intermediates[1] is not None else None
    dot_detection_image = encode_image(intermediates[2]) if intermediates[2] is not None else None
    straightened_dots_image = encode_image(intermediates[3]) if intermediates[3] is not None else None
    images = ImagePipeline(original_image=image_bytes, pose_estimation_image=pose_estimation_image, cropped_image=cropped_image, dot_detection_image=dot_detection_image, straightened_dots_image=straightened_dots_image)
    store_dataclass(images)
    sighting = Sighting(individual_id=None, image_id=images.id, coordinates=list(coordinates))
    sighting = store_dataclass(sighting)
    return {
        "sighting_id": sighting.id,
        "candidates": converted_list,
        "quality": quality.name
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

@app.route('/auth', methods=['POST'])
def auth():
    password = request.json["password"]
    hashed_passwords = db.session.query(Password).all()
    if not any(bcrypt.checkpw(password.encode('utf-8'), hashed_password.password.encode('utf-8')) for hashed_password in hashed_passwords):
        return Response(status=401)

    exp = datetime.utcnow() + timedelta(days=1)

    return jwt.encode({'exp': exp}, jwt_secret, algorithm='HS256')
@app.before_request
def check_auth():
    if (request.method not in ['GET', 'OPTIONS']) and request.path != '/auth':
        header = request.headers.get('Authorization')
        if header is None:
            return Response(status=401)
        header = header.replace('Bearer ', '')
        try:
            jwt.decode(header, jwt_secret, algorithms=['HS256'])
            print("success")
        except jwt.ExpiredSignatureError:
            return Response(status=401)
        except jwt.InvalidTokenError:
            return Response(status=401)

@app.after_request
def after_request(response):
    # cors stuff to make flutter-web work
    response.headers['Access-Control-Allow-Origin'] = request.origin
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE'
    response.headers['Vary'] = 'Origin'
    return response

@app.route("/")
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
