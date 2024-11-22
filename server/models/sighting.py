from server.models import Base
from sqlalchemy import Integer, Float, DateTime, ARRAY, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional, Tuple

class Sighting(Base):
    __tablename__ = 'sightings'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    individual_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    coordinates: Mapped[list[Tuple[float, float]]] = mapped_column(ARRAY(Float))
    date: Mapped[DateTime] = mapped_column(DateTime, nullable=True)
    image_id: Mapped[Optional[int]] = mapped_column(ForeignKey('image_pipeline.id'), nullable=True)
