"""This module defines a ZFP compression parameterization model using Pydantic.

Classes:
    ZFPMode: Enum that characterizes the ZFP compression mode.
    ZFP: Class that represents the ZFP compression model.

ZFPMode Enum Values:
    FIXED_ACCURACY: Allows fixed accuracy mode.
    FIXED_PRECISION: Allows fixed precision mode.
    FIXED_RATE: Allows fixed rate mode.
    REVERSIBLE: Allows reversible mode.

ZFP Class Attributes:
    mode: Specifies the compression mode.
    tolerance: Specifies the parameter for the FIXED_ACCURACY mode.
    rate: Specifies the parameter for the FIXED_RATE mode.
    precision: Specifies the parameter for the FIXED_PRECISION mode.
    write_header: Allows encoding of shape, scalar type, and compression parameters.

Notes:
    Only one parameter(tolerance, rate, precision) should be given at a time for a
    respective mode. For REVERSIBLE mode, no parameter (tolerance, rate, precision)
    should be used.
"""


from enum import StrEnum

from pydantic import Field
from pydantic import model_validator

from mdio.schemas.base.core import StrictCamelBaseModel


class ZFPMode(StrEnum):
    """Enum for ZFP algorithm modes."""

    FIXED_ACCURACY = "fixed_accuracy"
    FIXED_PRECISION = "fixed_precision"
    FIXED_RATE = "fixed_rate"
    REVERSIBLE = "reversible"


class ZFP(StrictCamelBaseModel):
    """Data Model for ZFP options."""

    name: str = Field(default="zfp", description="Name of the compressor.")
    mode: ZFPMode = Field()

    tolerance: float | None = Field(
        default=None,
        description="Fixed accuracy in terms of absolute error tolerance.",
    )

    rate: float | None = Field(
        default=None,
        description="Fixed rate in terms of number of compressed bits per value.",
    )

    precision: int | None = Field(
        default=None,
        description="Fixed precision in terms of number of uncompressed bits per value.",
    )

    write_header: bool = Field(
        default=True,
        description="Encode array shape, scalar type, and compression parameters.",
    )

    @model_validator(mode="after")
    def check_requirements(self):
        """Check if ZFP parameters make sense."""
        mode = self.mode

        if mode == ZFPMode.REVERSIBLE:
            if any(
                getattr(self, key) is not None
                for key in ["tolerance", "rate", "precision"]
            ):
                raise ValueError("Other fields must be None in REVERSIBLE mode")

        if mode == ZFPMode.FIXED_ACCURACY and self.tolerance is None:
            raise ValueError("tolerance required for FIXED_ACCURACY")
        elif mode == ZFPMode.FIXED_RATE and self.rate is None:
            raise ValueError("rate required for FIXED_RATE")
        elif mode == ZFPMode.FIXED_PRECISION and self.precision is None:
            raise ValueError("precision required for FIXED_PRECISION")

        return self
