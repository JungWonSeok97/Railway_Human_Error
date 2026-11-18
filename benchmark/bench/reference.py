"""CatBoost reference predictor (Railway_hep models, unchanged) and HEP calculation."""
from __future__ import annotations

import json
import math
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

from bench.spec import LEVELS, MULTIPLIERS, ROLE_BASE_HEP, ROOT, VENDOR

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(VENDOR))


def load_spec() -> dict:
    return json.loads((ROOT / "data" / "survey_spec.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _model(pid: int):
    from catboost import CatBoostClassifier
    from config.catboost_config import CATBOOST_CONFIG
    from config.paths import CATBOOST_MODEL_DIR

    m = CatBoostClassifier()
    m.load_model(str(Path(CATBOOST_MODEL_DIR) / CATBOOST_CONFIG[pid]["filename"]))
    return m


def features(pid: int, answers: dict[str, int]) -> list[str]:
    """answers: {"F.p.S.k": 0/1/2} -> CatBoost input row (GMM composites included)."""
    from preprocessing.catboost_features import build_catboost_features

    return build_catboost_features(pid, answers)


def predict(pid: int, answers_list: list[dict[str, int]]) -> tuple[list[str], np.ndarray]:
    """Return (level labels, class probabilities ordered like LEVELS[pid])."""
    m = _model(pid)
    rows = [features(pid, a) for a in answers_list]
    proba = m.predict_proba(rows)
    classes = [str(c) for c in m.classes_]
    order = [classes.index(lv) for lv in LEVELS[pid]]
    proba = proba[:, order]
    labels = [LEVELS[pid][int(i)] for i in proba.argmax(axis=1)]
    return labels, proba


def psf4_multiplier(months: float) -> float:
    n = months / 18.555
    frac = 0.0 if months == 0 else (n**4 / (1 + n**4) if n <= 1 else 1 / (1 + (1 / n) ** 4))
    return 0.33 + 4.5 * frac


def multiplier(levels: dict[int, str], months: float) -> float:
    """Product M of all PSF multipliers (F.4 from the elapsed-month formula)."""
    m = psf4_multiplier(months)
    for pid, lv in levels.items():
        if pid != 4:
            m *= MULTIPLIERS[pid][lv]
    return m


def hep(role: str, levels: dict[int, str], months: float) -> float:
    """Final HEP exactly as Railway_hep result.html computes it. Note: it can exceed 1 (up to 2), because the
    saturation-corrected diagnosis and execution parts are each below 1 but are added."""
    m = multiplier(levels, months)
    d, e = ROLE_BASE_HEP[role]
    final = d * m + e * m
    if final > 1:
        final = d * m / (d * (m - 1) + 1) + e * m / (e * (m - 1) + 1)
    return final


def log10_ratio(a: float, b: float) -> float:
    return abs(math.log10(a) - math.log10(b))
