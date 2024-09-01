from server.database_interface import db

db["individuals"].delete_many({})
db["sightings"].delete_many({})
db["fs.files"].delete_many({})
db["fs.chunks"].delete_many({})