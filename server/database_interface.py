from pymongo import MongoClient, CursorType

from server.config import MONGO_CONNECTION_STRING
client = MongoClient(MONGO_CONNECTION_STRING)

db = client["Salamanders"]
collection = db["individuals"]

def get_individuals_coords():
    cursor: CursorType = collection.find()
    # change the coords to a list of tuples to make it hashable
    # also remove the _id field
    list = [(doc["name"], [(coords[0], coords[1]) for coords in doc["coords"]]) for doc in cursor]
    return list

def get_individual_coords(name):
    doc = collection.find_one({"name": name})
    return [(coords[0], coords[1]) for coords in doc["coords"]]

