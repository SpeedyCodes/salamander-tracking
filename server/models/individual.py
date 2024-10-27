from server.models import Base
from sqlalchemy import Integer, String, ARRAY
from sqlalchemy.orm import Mapped, mapped_column


class Individual(Base):
    __tablename__ = 'individuals'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[String] = mapped_column(String(100))
    sighting_ids: Mapped[list[int]] = mapped_column(ARRAY(String))
