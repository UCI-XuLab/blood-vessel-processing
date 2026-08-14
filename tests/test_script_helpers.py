"""Tests for the shared helpers in scripts/.

`scripts/` had no tests. That was tolerable while it held only batch loops, but
`analyse_spinal_cord` now hosts the section loader and the CSV writer that every
other script routes through, and `write_csv` writes the three result CSVs that
are the actual deliverables - including the handoff's primary measure. Those are
worth pinning even though the analyses themselves cannot run here (the imaging
data lives on a lab share).

Only the data-independent helpers are covered: anything touching Z: is out of
reach from a clone.
"""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyse_spinal_cord import (Section, short_reporter, virus_cut,  # noqa: E402
                                 write_csv)


# --------------------------------------------------------------------------
# write_csv
# --------------------------------------------------------------------------

def test_write_csv_round_trips(tmp_path):
    rows = [{"mouse": "M131", "region": "C", "dice": 0.42},
            {"mouse": "M131", "region": "T", "dice": 0.51}]
    path = write_csv(rows, tmp_path / "out.csv")

    with open(path, newline="", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert [r["mouse"] for r in loaded] == ["M131", "M131"]
    assert [r["region"] for r in loaded] == ["C", "T"]
    assert float(loaded[1]["dice"]) == pytest.approx(0.51)


def test_write_csv_preserves_column_order(tmp_path):
    """Header order comes from the first row, so a reader sees the field order
    the script chose rather than something alphabetised."""
    path = write_csv([{"z": 1, "a": 2, "m": 3}], tmp_path / "order.csv")
    assert path.read_text(encoding="utf-8").splitlines()[0] == "z,a,m"


def test_write_csv_refuses_to_write_nothing(tmp_path):
    """An empty run must fail loudly, not leave a header-only file that reads
    as a successful analysis over zero sections."""
    target = tmp_path / "empty.csv"
    with pytest.raises(ValueError):
        write_csv([], target)
    assert not target.exists()


# --------------------------------------------------------------------------
# virus_cut
# --------------------------------------------------------------------------

def test_virus_cut_is_median_plus_k_mad():
    import numpy as np
    values = np.array([10.0, 10.0, 10.0, 10.0, 10.0])   # zero spread -> cut == median
    cut, background = virus_cut(values, k=3.0)
    assert background == pytest.approx(10.0)
    assert cut == pytest.approx(10.0)

    # 1.4826 * MAD estimates sigma, so a normal sample lands near median + k*sigma.
    rng = np.random.default_rng(0)
    sample = rng.normal(100.0, 5.0, size=20000)
    cut, background = virus_cut(sample, k=3.0)
    assert background == pytest.approx(100.0, abs=0.2)
    assert cut == pytest.approx(115.0, abs=0.5)


def test_virus_cut_is_robust_to_a_bright_minority():
    """The point of median + MAD: vessels are a bright minority and must not
    drag the parenchyma background up with them."""
    import numpy as np
    quiet = np.full(1000, 100.0)
    with_vessels = np.concatenate([quiet, np.full(100, 5000.0)])
    assert virus_cut(with_vessels)[1] == pytest.approx(virus_cut(quiet)[1])


# --------------------------------------------------------------------------
# Section
# --------------------------------------------------------------------------

def _section(**overrides):
    import numpy as np
    empty = np.zeros((2, 2), dtype=np.float32)
    fields = dict(path=Path("Fig 1_M131_C_SYFP2-green_CD31-mag_slice3.tif"),
                  index=7, total=39, figure="Fig 1", mouse="M131", region="C",
                  reporter="SYFP2", slice_id="3",
                  virus=empty, cd31=empty, tissue=empty.astype(bool))
    fields.update(overrides)
    return Section(**fields)


def test_section_formats_label_stem_and_counter():
    section = _section()
    assert section.label == "Fig 1 M131 cervical s3"
    assert section.stem == "Fig1_M131_C_SYFP2_s3"
    assert section.counter == "[ 7/39]"


def test_section_without_a_slice_number_still_gets_a_stem():
    """Gallery filenames always carry a slice field, `_s0` when the source has
    none, so a slice-less section cannot collide with a numbered one."""
    section = _section(slice_id="")
    assert section.label == "Fig 1 M131 cervical"
    assert section.stem.endswith("_s0")


def test_sections_are_usable_as_dict_keys():
    """eq=False on the dataclass: the generated __eq__/__hash__ would close over
    the ndarray fields and raise. Identity semantics keep a Section hashable."""
    a, b = _section(), _section()
    assert a != b
    assert len({a, b}) == 2
    assert {a: "first"}[a] == "first"


# --------------------------------------------------------------------------
# short_reporter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("SYFP2-green", "SYFP2"),
    ("SYFP2", "SYFP2"),
    ("tdTomato-red", "tdT"),
    ("tdT", "tdT"),
    ("something-else", "something-else"),
])
def test_short_reporter_maps_the_filename_field_to_its_display_form(raw, expected):
    assert short_reporter(raw) == expected
