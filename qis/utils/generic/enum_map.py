from enum import Enum


class EnumMap(Enum):
    """
    abstract enum with a map function
    """
    @classmethod
    def map_to_value(cls, name):
        """
        given name return value
        """
        for k, v in cls.__members__.items():
            if k == name:
                return v
        raise ValueError(f"nit in enum {name}")