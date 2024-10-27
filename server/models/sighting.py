from server.models import Base
from sqlalchemy import Integer, Float, DateTime, ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional, Tuple
from sqlalchemy.dialects.postgresql import BYTEA

class Sighting(Base):
    __tablename__ = 'sightings'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    individual_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    coordinates: Mapped[list[Tuple[float, float]]] = mapped_column(ARRAY(Float))
    date: Mapped[DateTime] = mapped_column(DateTime, nullable=True)
    image: Mapped[bytes] = mapped_column(BYTEA)
