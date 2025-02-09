from server.models import Base
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer


class Password(Base):
    __tablename__ = 'passwords'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    password: Mapped[String] = mapped_column(String(100))
