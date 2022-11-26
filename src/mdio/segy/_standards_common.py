"""Common elements for SEG-Y standards datasets."""


from enum import IntEnum


class SegyFloatFormat(IntEnum):
    """Numeric type to SEG-Y code mapping."""

    IBM32 = 1
    INT32 = 2
    INT16 = 3
    FLOAT32 = 5
    FLOAT64 = 6
    INT8 = 8
    INT64 = 9
    UINT32 = 10
    UINT16 = 11
    UINT64 = 12
    UINT8 = 16


class TraceSortingCode(IntEnum):
    """Data sorting type to SEG-Y code mapping."""

    NO_SORTING = 1
    CDP_ENSEMBLE = 2
    SINGLE_FOLD = 3
    HORZ_STACKED = 4


class SweepTypeCode(IntEnum):
    """Sweep type to SEG-Y code mapping."""

    LINEAR = 1
    PARABOLIC = 2
    EXPONENTIAL = 3
    OTHER = 4
