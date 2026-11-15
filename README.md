# Speed-Lib

[![CI](https://github.com/kapildhinwa2010-dev/Speed-Lib/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Speed-Lib/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/Speed-Lib/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Speed-Lib/actions/workflows/codeql.yml)

Reusable timing utilities: best-of-N wall-clock measurement for the toolkit codecs.

## What it does

- timed(func, repeats) runs a callable N times and reports min/avg/max
  wall-clock seconds. min_s is the honest number; avg and max show noise.
- method_timings(matrix, repeats) times all four toolkit codecs (SVD,
  quantize, Fourier, hybrid) on one matrix, sorted fastest first.

    from speed_lib import method_timings

    for row in method_timings(matrix, repeats=5):
        print(row["method"], round(row["min_s"] * 1000, 3), "ms")

## API

- timed(func, repeats=5) - dict with repeats, min_s, avg_s, max_s
- method_timings(matrix, repeats=3) - list of dicts, fastest first
- The four codecs are included so the workload is self-contained.

## Honest results and limitations

- Wall-clock timings on shared or cloud machines are noisy; never
  compare runs with different repeats or machine load. Report min-of-N.
- These are pure-numpy timings on CPU; they say nothing about GPU or
  production serving latency.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
