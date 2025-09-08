"""Integration tests data for teapot dome SEG-Y."""

from segy.schema import SegySpec
from segy.standards import get_segy_standard
from tests.integration.testing_helpers import customize_segy_specs


def custom_teapot_dome_segy_spec(keep_unaltered: bool) -> SegySpec:
    """Return the minimum customized SEG-Y specification for the teapot dome dataset.

    In SEG-Y spec rev 1.0:
        inline                      = (189, "int32")
        crossline                   = (193, "int32")
        cdp_x                       = (181, "int32")
        cdp_y                       = (185, "int32")
    and
        trace_num_orig_record       = (13, "int32")
        energy_source_point_num     = (17, "int32")
        group_coord_x               = (81, "int32")
        group_coord_y               = (85, "int32")

    In SEGY 1.0 - 2.1, the trace header contains unassigned bytes 181-240.
    """
    index_bytes: tuple[int, ...] = (17, 13, 81, 85)
    index_names: tuple[str, ...] = ("inline", "crossline", "cdp_x", "cdp_y")
    index_types: tuple[str, ...] = ("int32", "int32", "int32", "int32")
    segy_spec = get_segy_standard(1.0)
    return customize_segy_specs(
        segy_spec=segy_spec,
        index_bytes=index_bytes,
        index_names=index_names,
        index_types=index_types,
        keep_unaltered=keep_unaltered,
    )


def text_header_teapot_dome() -> list[str]:
    """Return the teapot dome expected text header."""
    return [
        "C 1 CLIENT: ROCKY MOUNTAIN OILFIELD TESTING CENTER                              ",
        "C 2 PROJECT: NAVAL PETROLEUM RESERVE #3 (TEAPOT DOME); NATRONA COUNTY, WYOMING  ",
        "C 3 LINE: 3D                                                                    ",
        "C 4                                                                             ",
        "C 5 THIS IS THE FILTERED POST STACK MIGRATION                                   ",
        "C 6                                                                             ",
        "C 7 INLINE 1, XLINE 1:   X COORDINATE: 788937  Y COORDINATE: 938845             ",
        "C 8 INLINE 1, XLINE 188: X COORDINATE: 809501  Y COORDINATE: 939333             ",
        "C 9 INLINE 188, XLINE 1: X COORDINATE: 788039  Y COORDINATE: 976674             ",
        "C10 INLINE NUMBER:    MIN: 1  MAX: 345  TOTAL: 345                              ",
        "C11 CROSSLINE NUMBER: MIN: 1  MAX: 188  TOTAL: 188                              ",
        "C12 TOTAL NUMBER OF CDPS: 64860   BIN DIMENSION: 110' X 110'                    ",
        "C13                                                                             ",
        "C14                                                                             ",
        "C15                                                                             ",
        "C16                                                                             ",
        "C17                                                                             ",
        "C18                                                                             ",
        "C19 GENERAL SEGY INFORMATION                                                    ",
        "C20 RECORD LENGHT (MS): 3000                                                    ",
        "C21 SAMPLE RATE (MS): 2.0                                                       ",
        "C22 DATA FORMAT: 4 BYTE IBM FLOATING POINT                                      ",
        "C23 BYTES  13- 16: CROSSLINE NUMBER (TRACE)                                     ",
        "C24 BYTES  17- 20: INLINE NUMBER (LINE)                                         ",
        "C25 BYTES  81- 84: CDP_X COORD                                                  ",
        "C26 BYTES  85- 88: CDP_Y COORD                                                  ",
        "C27 BYTES 181-184: INLINE NUMBER (LINE)                                         ",
        "C28 BYTES 185-188: CROSSLINE NUMBER (TRACE)                                     ",
        "C29 BYTES 189-192: CDP_X COORD                                                  ",
        "C30 BYTES 193-196: CDP_Y COORD                                                  ",
        "C31                                                                             ",
        "C32                                                                             ",
        "C33                                                                             ",
        "C34                                                                             ",
        "C35                                                                             ",
        "C36 Processed by: Excel Geophysical Services, Inc.                              ",
        "C37               8301 East Prentice Ave. Ste. 402                              ",
        "C38               Englewood, Colorado 80111                                     ",
        "C39               (voice) 303.694.9629 (fax) 303.771.1646                       ",
        "C40 END EBCDIC                                                                  ",
    ]


def binary_header_teapot_dome() -> dict[str, int]:
    """Return the teapot dome expected binary header."""
    return {
        "job_id": 9999,
        "line_num": 9999,
        "reel_num": 1,
        "data_traces_per_ensemble": 188,
        "aux_traces_per_ensemble": 0,
        "sample_interval": 2000,
        "orig_sample_interval": 0,
        "samples_per_trace": 1501,
        "orig_samples_per_trace": 1501,
        "data_sample_format": 1,
        "ensemble_fold": 57,
        "trace_sorting_code": 4,
        "vertical_sum_code": 1,
        "sweep_freq_start": 0,
        "sweep_freq_end": 0,
        "sweep_length": 0,
        "sweep_type_code": 0,
        "sweep_trace_num": 0,
        "sweep_taper_start": 0,
        "sweep_taper_end": 0,
        "taper_type_code": 0,
        "correlated_data_code": 2,
        "binary_gain_code": 1,
        "amp_recovery_code": 4,
        "measurement_system_code": 2,
        "impulse_polarity_code": 1,
        "vibratory_polarity_code": 0,
        "fixed_length_trace_flag": 0,
        "num_extended_text_headers": 0,
        "segy_revision_major": 0,
        "segy_revision_minor": 0,
    }
