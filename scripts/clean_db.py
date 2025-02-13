from server.app import app
from server.app import db



# drop all tables
with app.app_context():
    db.drop_all()
    db.create_all()
