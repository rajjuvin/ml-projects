import tomllib
from pathlib import Path

from readmission import __version__

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_the_version_is_written_in_one_place_only():
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert "version" not in config["project"] and "version" in config["project"]["dynamic"]
    assert config["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "readmission.__version__"}
    assert __version__.count(".") == 2  # the release workflow compares the tag v<version> with this value
