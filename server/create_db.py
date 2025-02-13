from sqlalchemy import create_engine
from sqlalchemy.schema import CreateTable

from models import *
from models.individual import Individual
from models.sighting import Sighting

path="postgresql://postgres:your_password@localhost:5432/salamanders"
engine = create_engine(path, echo=True)
print(CreateTable(Sighting.__table__).compile(engine))
#Base.metadata.create_all(engine)
