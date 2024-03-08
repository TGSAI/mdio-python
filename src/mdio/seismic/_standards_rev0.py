"""SEG-Y Rev 0 standard and its definitions."""

from mdio.seismic.byte_utils import ByteOrder
from mdio.seismic.byte_utils import Dtype
from mdio.seismic.byte_utils import OrderedType
from mdio.seismic.headers import Header
from mdio.seismic.headers import HeaderGroup

# ruff: noqa: E501

SEGY_REV0_TEXT = {"rows": 40, "cols": 80, "word_length": 4}

# fmt: off
SEGY_REV0_BINARY_HEADER = [
    Header(name="JobIdentificationNumber", offset=0, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="LineNumber", offset=4, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="ReelNumber", offset=8, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="NumTracesPerRecord", offset=12, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="NumAuxTracesPerRecord", offset=14, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SampleRate", offset=16, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SampleRateOriginal", offset=18, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="NumSamples", offset=20, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="NumSamplesOriginal", offset=22, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SampleFormat", offset=24, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="CdpFold", offset=26, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="Sorting", offset=28, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="VerticalSum", offset=30, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepFreqStart", offset=32, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepFreqEnd", offset=34, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepLength", offset=36, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepType", offset=38, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepTraceNumber", offset=40, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepTaperStartLen", offset=42, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepTaperEndLen", offset=44, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepTaperType", offset=46, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="Correlated", offset=48, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="BinaryGainRecovered", offset=50, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="AmplitudeRecoveryMethod", offset=52, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="MeasurementSystem", offset=54, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="ImpulseSignal", offset=56, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="VibratorPolarity", offset=58, type=Dtype.INT16, endian=ByteOrder.BIG),
]  # fmt: on

# fmt: off
SEGY_REV0_TRACE_HEADER = [
    Header(name="TraceSequenceNumberLine", offset=0, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="TraceSequenceNumberReel", offset=4, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="OriginalFieldRecordNumber", offset=8, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="TraceNumberOriginalFieldRecord", offset=12, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="EnergySourcePointNumber", offset=16, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="CdpEnsembleNumber", offset=20, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="TraceNumberWithinCdp", offset=24, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="TraceIdentification", offset=28, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="NumVerticallySummedTraces", offset=30, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="NumHorizontallyStackedTraces", offset=32, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="DataUse", offset=34, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SourceReceiverDistance", offset=36, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="ReceiverGroupElevation", offset=40, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="SourceElevation", offset=44, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="SourceDepth", offset=48, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="ReceiverDatumElevation", offset=52, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="SourceDatumElevation", offset=56, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="SourceWaterDepth", offset=60, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="ReceiverGroupWaterDepth", offset=64, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="ScalerElevationDepth", offset=68, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="ScalerCoordinates", offset=70, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SourceCoordinateX", offset=72, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="SourceCoordinateY", offset=76, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="GroupCoordinateX", offset=80, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="GroupCoordinateY", offset=84, type=Dtype.INT32, endian=ByteOrder.BIG),
    Header(name="CoordinateUnits", offset=88, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="WeatheringVelocity", offset=90, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SubWeatheringVelocity", offset=92, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SourceUpholeTime", offset=94, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="GroupUpholeTime", offset=96, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SourceStaticCorrection", offset=98, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="GroupStaticCorrection", offset=100, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="TotalStatic", offset=102, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="LagTimeA", offset=104, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="LagTimeB", offset=106, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="DelayRecordingTime", offset=108, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="BruteTimeStart", offset=110, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="MuteTimeEnd", offset=112, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SampleRateTrace", offset=114, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="NumSamplesTrace", offset=116, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="GainTypeField", offset=118, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="InstrumentGainConst", offset=120, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="InstrumentInitGain", offset=122, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="Correlated", offset=124, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepFreqStart", offset=126, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepFreqEnd", offset=128, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepLength", offset=130, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepType", offset=132, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepTaperStartLen", offset=134, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepTaperEndLen", offset=136, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="SweepTaperType", offset=138, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="AliasFilterFreq", offset=140, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="AliasFilterSlope", offset=142, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="NotchFilterFreq", offset=144, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="NotchFilterSlope", offset=146, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="LowCutFreq", offset=148, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="HighCutFreq", offset=150, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="LowCutSlope", offset=152, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="HighCutSlope", offset=154, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="Year", offset=156, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="Day", offset=158, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="Hour", offset=160, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="Minute", offset=162, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="Second", offset=164, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="TimeBasis", offset=166, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="TraceWeightingFactor", offset=168, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="GeophoneGroupNumberOfRoll", offset=170, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="GroupNumberOfFirstTrace", offset=172, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="GroupNumberOfLastTrace", offset=174, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="GapSize", offset=176, type=Dtype.INT16, endian=ByteOrder.BIG),
    Header(name="OvertravelTaper", offset=178, type=Dtype.INT16, endian=ByteOrder.BIG),
]  # fmt: on

# fmt: off
SEGY_REV0 = {
    "text_header": SEGY_REV0_TEXT,
    "binary_header": HeaderGroup(name='BinaryHeader', offset=3200, itemsize=400, headers=SEGY_REV0_BINARY_HEADER),
    "trace_header": HeaderGroup(name='TraceHeader', offset=3600, itemsize=240, headers=SEGY_REV0_TRACE_HEADER),
    "trace_format": OrderedType(Dtype.IBM32, ByteOrder.BIG),
}  # fmt: on
