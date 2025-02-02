from server.models import Base
from sqlalchemy import Integer, String, ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import BYTEA


class ImagePipeline(Base):
    __tablename__ = 'image_pipelines'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)
    original_image: Mapped[bytes] = mapped_column(BYTEA)
    pose_estimation_image: Mapped[bytes] = mapped_column(BYTEA, nullable=True)
    cropped_image: Mapped[bytes] = mapped_column(BYTEA, nullable=True)
    dot_detection_image: Mapped[bytes] = mapped_column(BYTEA, nullable=True)
    straightened_dots_image: Mapped[bytes] = mapped_column(BYTEA, nullable=True)
