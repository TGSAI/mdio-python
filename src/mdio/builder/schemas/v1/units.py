"""Unit schemas specific to MDIO v1."""

from __future__ import annotations

from typing import TypeAlias

from pint import UnitRegistry

from mdio.builder.schemas.units import UnitEnum
from mdio.builder.schemas.units import create_unit_model

ureg = UnitRegistry()
ureg.formatter.default_format = "~C"  # compact, abbreviated (symbol).


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


LengthUnitModel = create_unit_model(LengthUnitEnum, "LengthUnitModel", "length", __name__)


class TimeUnitEnum(UnitEnum):
    """Enum class representing units of time."""

    NANOSECOND = ureg.nanosecond
    MICROSECOND = ureg.microsecond
    MILLISECOND = ureg.millisecond
    SECOND = ureg.second
    MINUTE = ureg.minute
    HOUR = ureg.hour
    DAY = ureg.day


TimeUnitModel = create_unit_model(TimeUnitEnum, "TimeUnitModel", "time", __name__)


class DensityUnitEnum(UnitEnum):
    """Enum class representing units of density."""

    GRAMS_PER_CC = ureg.gram / ureg.centimeter**3
    KILOGRAMS_PER_M3 = ureg.kilogram / ureg.meter**3
    POUNDS_PER_GAL = ureg.pounds / ureg.gallon


DensityUnitModel = create_unit_model(DensityUnitEnum, "DensityUnitModel", "density", __name__)


class SpeedUnitEnum(UnitEnum):
    """Enum class representing units of speed."""

    METERS_PER_SECOND = ureg.meter / ureg.second
    FEET_PER_SECOND = ureg.feet / ureg.second


SpeedUnitModel = create_unit_model(SpeedUnitEnum, "SpeedUnitModel", "speed", __name__)


class AngleUnitEnum(UnitEnum):
    """Enum class representing units of angle."""

    DEGREES = ureg.degree
    RADIANS = ureg.radian


AngleUnitModel = create_unit_model(AngleUnitEnum, "AngleUnitModel", "angle", __name__)


class FrequencyUnitEnum(UnitEnum):
    """Enum class representing units of frequency."""

    HERTZ = ureg.hertz


FrequencyUnitModel = create_unit_model(FrequencyUnitEnum, "FrequencyUnitModel", "frequency", __name__)


class VoltageUnitEnum(UnitEnum):
    """Enum class representing units of voltage."""

    MICROVOLT = ureg.microvolt
    MILLIVOLT = ureg.millivolt
    VOLT = ureg.volt


VoltageUnitModel = create_unit_model(VoltageUnitEnum, "VoltageUnitModel", "voltage", __name__)


# Composite model types
AllUnitModel: TypeAlias = (
    LengthUnitModel
    | TimeUnitModel
    | AngleUnitModel
    | DensityUnitModel
    | SpeedUnitModel
    | FrequencyUnitModel
    | VoltageUnitModel
)
