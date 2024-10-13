from src.utils.heic_imread_wrapper import wrapped_imread
from enum import IntEnum


class ImageQuality(IntEnum):
    BAD = 1
    MEDIUM = 2
    GOOD = 3
