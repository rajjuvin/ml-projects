"""Download the UCI "Diabetes 130-US Hospitals for Years 1999-2008" dataset into data/.

    python scripts/download_data.py

The files are not committed to the repository. The download is verified against the SHA-256 of the
original file used to develop and evaluate the models, so results are reproducible.

Dataset: Clore et al. (2014), https://doi.org/10.24432/C5230J, licensed CC BY 4.0.
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXPECTED_SHA256 = {
    "diabetic_data.csv": "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97",
    "IDS_mapping.csv": "f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e",
}


def main():
    DATA_DIR.mkdir(exist_ok=True)
    if all((DATA_DIR / name).exists() for name in EXPECTED_SHA256):
        print(f"Data already present in {DATA_DIR}")
        return 0

    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    for name, expected in EXPECTED_SHA256.items():
        content = archive.read(name)
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            print(f"Checksum mismatch for {name}: expected {expected}, got {actual}", file=sys.stderr)
            return 1
        (DATA_DIR / name).write_bytes(content)
        print(f"Saved {DATA_DIR / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
