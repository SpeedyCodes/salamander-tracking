from server.models import Base
from sqlalchemy import Integer, String, ARRAY
from sqlalchemy.orm import Mapped, mapped_column


class Individual(Base):
    __tablename__ = 'individuals'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    name: Mapped[String] = mapped_column(String(100))
