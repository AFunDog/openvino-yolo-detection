"""Small GUI-related utilities."""

from __future__ import annotations

import json
import os


def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
