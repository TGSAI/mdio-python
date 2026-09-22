"""A module for converting numpy dtypes to MDIO scalar and structured types."""

from numpy import dtype as np_dtype

from mdio.builder.schemas.dtype import FixedStringType
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredField
from mdio.builder.schemas.dtype import StructuredType
from mdio.builder.schemas.dtype import is_fixed_string


def to_scalar_type(data_type: np_dtype) -> ScalarType | FixedStringType:
    """Convert numpy dtype to an MDIO ``ScalarType`` or ``FixedStringType``.

    Out of the 24 built-in numpy scalar type objects
    (see https://numpy.org/doc/stable/reference/arrays.dtypes.html)
    this function supports only a limited subset:
        ScalarType.INT8 <-> int8
        ScalarType.INT16 <-> int16
        ScalarType.INT32 <-> int32
        ScalarType.INT64 <-> int64
        ScalarType.UINT8 <-> uint8
        ScalarType.UINT16 <-> uint16
        ScalarType.UINT32 <-> uint32
        ScalarType.UINT64 <-> uint64
        ScalarType.FLOAT32 <-> float32
        ScalarType.FLOAT64 <-> float64
        ScalarType.COMPLEX64 <-> complex64
        ScalarType.COMPLEX128 <-> complex128
        ScalarType.BOOL <-> bool

    ``FixedStringType`` is accepted at any width. Numpy names ``|S<n>`` as
    ``bytes<8n>`` and ``<U<n>`` as ``str<32n>``, and those names are not valid
    ``np.dtype`` constructors, so ``FixedStringType`` uses the constructor form
    ``S<n>`` or ``U<n>``.

    Args:
        data_type: numpy dtype to convert

    Returns:
        ``ScalarType`` or ``FixedStringType``.

    Raises:
        ValueError: if dtype is not supported
    """
    # dtype.str is '<U8' / '|S8'. The tail is the canonical constructor token.
    token = data_type.str[1:]
    if is_fixed_string(token):
        return token

    try:
        return ScalarType(data_type.name)
    except ValueError as exc:
        err = f"Unsupported numpy dtype '{data_type.name}' for conversion to ScalarType."
        raise ValueError(err) from exc


def to_structured_type(data_type: np_dtype) -> StructuredType:
    """Convert numpy dtype to MDIO StructuredType.

    This function supports only a limited subset of structured types.
    In particular:
    It does not support nested structured types.
    Field types are the scalar subset from `to_scalar_type`, including ``FixedStringType``.

    Args:
        data_type: numpy dtype to convert

    Returns:
        StructuredType: corresponding MDIO structured type

    Raises:
        ValueError: if dtype is not structured or has no fields

    """
    if data_type is None or len(data_type.names or []) == 0:
        err = "None or empty dtype provided, cannot convert to StructuredType."
        raise ValueError(err)

    fields = []
    for field_name in data_type.names:
        field_dtype = data_type.fields[field_name][0]
        scalar_type = to_scalar_type(field_dtype)
        structured_field = StructuredField(name=field_name, format=scalar_type)
        fields.append(structured_field)
    return StructuredType(fields=fields)


def to_numpy_dtype(data_type: ScalarType | FixedStringType | StructuredType) -> np_dtype:
    """Get the numpy dtype for a variable."""
    if isinstance(data_type, StructuredType):
        return np_dtype([(field.name, to_numpy_dtype(field.format)) for field in data_type.fields])
    if isinstance(data_type, ScalarType):
        return np_dtype(data_type.value)
    if is_fixed_string(data_type):
        return np_dtype(data_type)
    msg = f"Expected ScalarType, FixedStringType, or StructuredType, got '{type(data_type).__name__}'"
    raise ValueError(msg)
