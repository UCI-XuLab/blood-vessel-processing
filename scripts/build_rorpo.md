# Building the reference RORPO

`scripts/path_operator.py` is a RORPO-family stand-in that runs with no build.
For the reference results, build the real thing on a machine with a C++
toolchain — this sandbox has neither cmake nor a compiler, so it cannot.

Source: https://github.com/path-openings/RORPO
Paper: Merveille et al., "Curvilinear Structure Analysis by Ranking the
Orientation Responses of Path Operators", IEEE TPAMI 2018.

## Prerequisites

- CMake ≥ 3.5
- A C++ compiler (MSVC on Windows, g++/clang elsewhere)
- Python 3 with numpy, for the `pyRORPO` bindings

On Windows the simplest toolchain is the "Build Tools for Visual Studio"
(the C++ workload), plus `pip install cmake`.

## Build

```bash
git clone https://github.com/path-openings/RORPO.git
cd RORPO
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPYTHON_BINDINGS=ON
cmake --build build --config Release
```

The `pyRORPO` module is written to `build/`. Put that directory on
`PYTHONPATH`, or copy the built module next to the scripts.

## Run it against the same sections

```python
import pyRORPO
# RORPO_multiscale(image, scale_min, factor, n_scales, ...) -> response
response = pyRORPO.RORPO_multiscale(cd31.astype("float32"),
                                    scaleMin=2, factor=1.5, nbScales=4)
```

Feed `response` into the same hysteresis + clean-up as the Jerman path
(`vessel_utils.threshold.segment`) so the two filters are compared on identical
post-processing, and score both against the REAVER manual ground-truth images
so the comparison rests on real accuracy rather than either filter's own output.
