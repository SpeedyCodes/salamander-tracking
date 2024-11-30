from flask_sqlalchemy import SQLAlchemy
from server.models import Base

db = SQLAlchemy(model_class=Base)

