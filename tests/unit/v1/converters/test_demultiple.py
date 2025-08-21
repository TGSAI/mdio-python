
import math
import random
import time
import numpy as np

from mdio.core.demultiple import demultiple_fast
from mdio.core.demultiple import demultiple_min_slow
from mdio.core.dimension import Dimension


def test_demultiple_min(capsys) -> None:
    dtype=[('inline', np.int32), ('crossline', np.int32), ('offset', np.int32), ('azimuth', np.int32), ('cdp_x', np.int32), ('cdp_y', np.int32)]

    do_timing = True
    if do_timing:
        # Use these values for performance estimates
        # At 100 x 100:
        #   Elapsed time (demultiple_fast): 0.001687 sec
        #   Elapsed time (demultiple_min_slow): 0.114320 sec
        # At 200 x 200:
        #   Elapsed time (demultiple_fast): 0.008996 sec
        #   Elapsed time (demultiple_min_slow): 0.471114 sec
        # At 1000 x 1000:
        #   Elapsed time (demultiple_fast): 0.251512 sec
        #   Elapsed time (demultiple_min_slow): 11.770302 sec
        inlines = np.array(range(1, 200), dtype=np.int32)
        crosslines = np.array(range(1, 200), dtype=np.int32)
    else:
        # Use these values for debugging
        inlines = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        crosslines = np.array([1, 2, 3, 4], dtype=np.int32)

    dims = [
        Dimension(inlines, "inline"),
        Dimension(crosslines, "crossline")
    ]

    inlines_data = np.random.permutation(inlines)
    crosslines_data = np.random.permutation(crosslines)
    offsets = np.random.permutation(np.array([0, 1, 2], dtype=np.int32))
    azimuths = np.random.permutation(np.array([0, 1], dtype=np.int32))

    data = []
    for il in inlines_data:
        for xl in crosslines_data:
            add_rand = 0
            for of in offsets:
                for az in azimuths:
                    x = 1000 * il + add_rand * random.randint(0, 10)
                    y = 1000 * xl + add_rand * random.randint(0, 10)
                    data.append((il, xl, of, az, x, y))
                    add_rand = 1
    struct_data = np.array(data, dtype=dtype)

    expected = np.array(
        [[1000, 1000, 1000, 1000],
        [2000, 2000, 2000, 2000],
        [3000, 3000, 3000, 3000],
        [4000, 4000, 4000, 4000],
        [5000, 5000, 5000, 5000]], dtype=np.int32)

    start_time = time.perf_counter()
    actual, delta = demultiple_fast(struct_data, dims, 'cdp_x')
    elapsed_time_one = time.perf_counter() - start_time
    with capsys.disabled():
        print("")
        print(f"Elapsed time (demultiple_fast): {elapsed_time_one:.6f} sec")
    if not do_timing:
        assert np.array_equal(actual, expected)
        assert delta <= 10, "Delta should be less than the maximum possible difference"

    start_time = time.perf_counter()
    actual = demultiple_min_slow(struct_data, dims, 'cdp_x')
    elapsed_time_two = time.perf_counter() - start_time
    with capsys.disabled():
        print(f"Elapsed time (demultiple_min_slow): {elapsed_time_two:.6f} sec")
    if not do_timing:
        assert np.array_equal(actual, expected)
