"""Unit schemas specific to MDIO v1."""


from typing import TypeAlias

from pint import UnitRegistry
from pydantic import Field

from mdio.schemas.base.metadata import VersionedMetadataConvention
from mdio.schemas.base.units import UnitEnum
from mdio.schemas.base.units import create_unit_model


ureg = UnitRegistry()
ureg.default_format = "~C"  # compact, abbreviated (symbol).


class LengthUnitEnum(UnitEnum):
    """Enum class representing metric units of length."""

    MILLIMETER = ureg.millimeter
    CENTIMETER = ureg.centimeter
    METER = ureg.meter
    KILOMETER = ureg.kilometer

    INCH = ureg.inch
    FOOT = ureg.foot
    YARD = ureg.yard
    MILE = ureg.mile


LengthUnitModel = create_unit_model(LengthUnitEnum, "LengthUnitModel", "length")


class TimeUnitEnum(UnitEnum):
    """Enum class representing units of time."""

    NANOSECOND = ureg.nanosecond
    MICROSECOND = ureg.microsecond
    MILLISECOND = ureg.millisecond
    SECOND = ureg.second
    MINUTE = ureg.minute
    HOUR = ureg.hour
    DAY = ureg.day


TimeUnitModel = create_unit_model(TimeUnitEnum, "TimeUnitModel", "time")


class DensityUnitEnum(UnitEnum):
    """Enum class representing units of density."""

    GRAMS_PER_CC = ureg.gram / ureg.centimeter**3
    KILOGRAMS_PER_CC = ureg.kilogram / ureg.meter**3
    POUNDS_PER_GAL = ureg.pounds / ureg.gallon


DensityUnitModel = create_unit_model(DensityUnitEnum, "DensityUnitModel", "density")


class SpeedUnitEnum(UnitEnum):
    """Enum class representing units of speed."""

    METER_PER_SECOND = ureg.meter / ureg.second
    FEET_PER_SECOND = ureg.feet / ureg.second


SpeedUnitModel = create_unit_model(SpeedUnitEnum, "SpeedUnitModel", "speed")


class AngleUnitEnum(UnitEnum):
    """Enum class representing units of angle."""

    DEGREES = ureg.degree
    RADIANS = ureg.radian


AngleUnitModel = create_unit_model(AngleUnitEnum, "AngleUnitModel", "angle")


class FrequencyUnitEnum(UnitEnum):
    """Enum class representing units of frequency."""

    FREQUENCY = ureg.hertz
    PHASE_DEG = ureg.degree
    PHASE_RAD = ureg.radian


FrequencyUnitModel = create_unit_model(
    FrequencyUnitEnum, "FrequencyUnitModel", "frequency"
)


# Composite model types
CoordinateUnitModel: TypeAlias = LengthUnitModel | TimeUnitModel | AngleUnitModel
AllUnitModel: TypeAlias = (
    CoordinateUnitModel | DensityUnitModel | SpeedUnitModel | FrequencyUnitModel
)


# Versioned metadata conventions for units
class CoordinateUnits(VersionedMetadataConvention):
    """Coordinate Units."""

    units_v1: CoordinateUnitModel = Field(...)


class AllUnits(VersionedMetadataConvention):
    """All Units."""

    units_v1: AllUnitModel | list[AllUnitModel] = Field(...)
