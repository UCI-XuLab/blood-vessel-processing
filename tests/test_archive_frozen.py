"""The published implementation must not drift.

`velazquez_rivera_2025` produced the numbers in the paper. Improvements belong in
`vessel_utils`; this package only changes if the published result is being
corrected, which is a decision someone should make deliberately rather than
discover in a diff.

If this test fails you either edited the archive by accident — revert — or you
meant to, in which case update MANIFEST below and say why in the commit message.
"""

import hashlib
from pathlib import Path

import pytest

ARCHIVE = Path(__file__).resolve().parent.parent / "archive" / "velazquez_rivera_2025"

MANIFEST = {
    "__init__.py": "9e1649c9560ce01886a1820759725138a6a5adefd6c02b19827e366667848d25",
    "enhance.py": "22a9bc63de625498a9f90838d1220303f5fe79b97b63594b1dd54846c41109e2",
    "io.py": "de1f4dac07efee8fef049d3aa0842c4f9294aa538055779695f653d86eac82fb",
    "metrics.py": "155998aab39cc4d206d46c3839b9c8c0f96d98fd35a940e13397cb37bb76ac79",
    "vessels.py": "afc92d36d8013d596785c8efbf0e9935f5b0411b32585cc441acce5dff476e59",
    "viz.py": "77c46aaebb757574061b965204991348685031c805514a2e3dcd78b420a34f83",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_archive_file_is_unchanged(name):
    path = ARCHIVE / name
    assert path.exists(), f"{name} is missing from the archive"
    assert digest(path) == MANIFEST[name], (
        f"{name} has changed. The published implementation is frozen — put the "
        f"improvement in vessel_utils/ instead. If the change is deliberate, "
        f"update MANIFEST in {Path(__file__).name} and explain it in the commit."
    )


def test_no_files_added_to_the_archive():
    present = {p.name for p in ARCHIVE.glob("*.py")}
    assert present == set(MANIFEST), (
        f"archive contents changed: added {sorted(present - set(MANIFEST))}, "
        f"removed {sorted(set(MANIFEST) - present)}"
    )


def test_archive_and_active_package_are_separate():
    """The archive must not import from the evolving package, or it would drift."""
    for path in ARCHIVE.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "vessel_utils" not in source, (
            f"{path.name} references vessel_utils; the archive must stay "
            f"self-contained so that changes there cannot alter published results"
        )
