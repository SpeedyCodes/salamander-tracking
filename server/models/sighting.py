from server.models import Base
from sqlalchemy import Integer, Float, DateTime, ARRAY, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional, Tuple
from sqlalchemy.orm import relationship

from server.models.individual import Individual
from server.models.named_location import NamedLocation


class Sighting(Base):
    __tablename__ = 'sightings'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    individual_id: Mapped[Optional[int]] = mapped_column(ForeignKey('individuals.id'), nullable=True)
    individual: Mapped["Individual"] = relationship(init=False)
    coordinates: Mapped[list[Tuple[float, float]]] = mapped_column(ARRAY(Float))
    date: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True, init=False)
    image_id: Mapped[Optional[int]] = mapped_column(ForeignKey('image_pipelines.id'), nullable=True) # TODO cascade delete
    location_id: Mapped[Optional[int]] = mapped_column(ForeignKey('named_locations.id'), nullable=True, init=False)
    location: Mapped["NamedLocation"] = relationship(init=False)
    precise_location: Mapped[Optional[Tuple[float, float]]] = mapped_column(ARRAY(Float), nullable=True, init=False)
