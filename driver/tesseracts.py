# SPDX-License-Identifier: Apache-2.0
"""Load the project Tesseracts from their API modules for the local dev loop."""

from pathlib import Path

import yaml
from tesseract_core import Tesseract

TESSERACT_ROOT = Path(__file__).resolve().parents[1] / "tesseracts"
NAMES = ("geom", "cfd", "em")


def local_tesseract(name: str) -> Tesseract:
    return Tesseract.from_tesseract_api(TESSERACT_ROOT / name / "tesseract_api.py")


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
