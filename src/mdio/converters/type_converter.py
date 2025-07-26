from numpy import dtype as np_dtype
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredType
from mdio.schemas.dtype import StructuredField
from mdio.schemas.v1.dataset_builder import _to_dictionary
from segy.arrays import HeaderArray as HeaderArray

def _to_scalar_type_using_kind_and_size(field_dtype: np_dtype, field_name: str) -> ScalarType:
        # Try to infer from numpy dtype kind and itemsize
    if field_dtype.kind == 'i':  # signed integer
        if field_dtype.itemsize == 1:
            return ScalarType.INT8
        elif field_dtype.itemsize == 2:
            return ScalarType.INT16
        elif field_dtype.itemsize == 4:
            return ScalarType.INT32
        elif field_dtype.itemsize == 8:
            return ScalarType.INT64
        else:
            raise ValueError(f"Unsupported integer size: {field_dtype.itemsize} bytes for field '{field_name}'")
    elif field_dtype.kind == 'u':  # unsigned integer
        if field_dtype.itemsize == 1:
            return ScalarType.UINT8
        elif field_dtype.itemsize == 2:
            return ScalarType.UINT16
        elif field_dtype.itemsize == 4:
            return ScalarType.UINT32
        elif field_dtype.itemsize == 8:
            return ScalarType.UINT64
        else:
            raise ValueError(f"Unsupported unsigned integer size: {field_dtype.itemsize} bytes for field '{field_name}'")
    elif field_dtype.kind == 'f':  # floating point
        if field_dtype.itemsize == 2:
            return ScalarType.FLOAT16
        elif field_dtype.itemsize == 4:
            return ScalarType.FLOAT32
        elif field_dtype.itemsize == 8:
            return ScalarType.FLOAT64
        else:
            raise ValueError(f"Unsupported float size: {field_dtype.itemsize} bytes for field '{field_name}'")
    elif field_dtype.kind == 'c':  # complex
        if field_dtype.itemsize == 8:
            return ScalarType.COMPLEX64
        elif field_dtype.itemsize == 16:
            return ScalarType.COMPLEX128
        else:
            raise ValueError(f"Unsupported complex size: {field_dtype.itemsize} bytes for field '{field_name}'")
    elif field_dtype.kind == 'b':  # boolean
        return ScalarType.BOOL
    else:
        raise ValueError(f"Unsupported numpy dtype: {field_dtype} for field '{field_name}'")

def to_scalar_type(field_dtype: np_dtype, field_name: str) -> ScalarType:
    """Convert numpy dtype to MDIO ScalarType.
    
    Args:
        field_dtype: numpy dtype to convert
        field_name: field name for error reporting
        
    Returns:
        ScalarType: corresponding MDIO scalar type
        
    Raises:
        ValueError: if dtype is not supported
    """

    numpy_type_str = field_dtype.name
    
    # Handle byte order specifications (remove endianness indicators)
    if numpy_type_str.startswith(('<', '>', '=')):
        numpy_type_str = numpy_type_str[1:]
    
    # Map numpy dtype to MDIO ScalarType

    # Direct mapping using ScalarType enum values
    try:
        return ScalarType(numpy_type_str)
    except ValueError:
        pass  # Fall back to inference logic

    return _to_scalar_type_using_kind_and_size(field_dtype, field_name)


# def to_structured_type(index_headers: HeaderArray) -> StructuredType:

def to_structured_type(dtype: np_dtype) -> StructuredType:

    if dtype is None or len(dtype.names or []) == 0:
        return None  # No headers to convert
    
    fields = []
    
    # Iterate through the structured array's field names and types
    for field_name in dtype.names:
        field_dtype = dtype.fields[field_name][0]

        # Convert numpy dtype to MDIO ScalarType
        scalar_type = to_scalar_type(field_dtype, field_name)

        # Create StructuredField and add to list
        structured_field = StructuredField(name=field_name, format=scalar_type)
        fields.append(structured_field)
    
    return StructuredType(fields=fields)


def to_numpy_dtype(data_type: ScalarType | StructuredType) -> np_dtype:
    """Get the numpy dtype for a variable."""
    if isinstance(data_type, ScalarType):
        return np_dtype(data_type.value)
    if isinstance(data_type, StructuredType):
        return np_dtype([(f.name, f.format.value) for f in data_type.fields])
    msg = f"Expected ScalarType or StructuredType, got '{type(data_type).__name__}'"
    raise ValueError(msg)
