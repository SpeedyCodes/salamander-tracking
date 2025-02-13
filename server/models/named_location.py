from server.models import Base
from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional, Tuple
from sqlalchemy import Integer, Float, DateTime, ARRAY


class NamedLocation(Base):
    __tablename__ = 'named_locations'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    name: Mapped[String] = mapped_column(String(100))
    precise_location: Mapped[Optional[Tuple[float, float]]] = mapped_column(ARRAY(Float), nullable=False)

