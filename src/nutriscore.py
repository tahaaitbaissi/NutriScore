"""Reference Nutri-Score calculator (general-food algorithm, 2023 and 2021).

Part 1 deliverable — EMSI Data Mining, « Retrouver les règles du Nutri-Score ».

The module is intentionally dependency-light (stdlib + PyYAML) so that Part 2
can import it without pulling pandas/scikit-learn.

Usage
-----
>>> from src.nutriscore import compute, score, grade
>>> x = {"energy_kj": 1703, "sat_fat": 3.1, "sugars": 25, "salt": 0.83,
...      "fibre": 3.8, "proteins": 8.0, "fvl_pct": 0}
>>> compute(x, version="2023")["score"]
18
>>> grade(18, "2023")
'D'

Conventions
-----------
* Nutrients are per 100 g. ``energy_kj`` in kJ, ``sat_fat``/``sugars``/``salt``/
  ``fibre``/``proteins`` in g, ``fvl_pct`` in % (0-100).
* Point convention: 0 point if value <= first threshold, k points if value is
  strictly greater than the k-th threshold.
* ``salt`` is always the input column: the 2021 algorithm converts it to
  sodium with ``sodium_mg = salt_g * 400`` (factor stored in the YAML).
* Thresholds live in ``config/thresholds.yaml`` and are loaded lazily; edit the
  YAML, not this file.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Iterable, Mapping

import yaml

__all__ = [
    "FEATURES",
    "VERSIONS",
    "DEFAULT_THRESHOLDS",
    "load_thresholds",
    "points",
    "fvl_points",
    "compute",
    "score",
    "grade",
]

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_THRESHOLDS = ROOT / "config" / "thresholds.yaml"

#: Columns the calculator consumes, in the order of the data contract.
FEATURES = ("energy_kj", "sat_fat", "sugars", "salt", "fibre", "proteins", "fvl_pct")

VERSIONS = ("2021", "2023")

_CACHE: dict[str, dict] = {}


def load_thresholds(path: str | Path | None = None) -> dict:
    """Load the ``versions`` mapping from ``config/thresholds.yaml``.

    Results are cached per path so that calling the calculator on 15 000 rows
    does not re-read the YAML each time.
    """
    key = str(path or DEFAULT_THRESHOLDS)
    if key not in _CACHE:
        with open(key, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        _CACHE[key] = doc["versions"]
    return _CACHE[key]


def _check_version(version: str) -> None:
    if version not in VERSIONS:
        raise ValueError(f"unknown Nutri-Score version {version!r}; expected one of {VERSIONS}")


def points(value: float, thresholds: Iterable[float]) -> int:
    """Number of thresholds strictly below ``value``.

    This is exactly the official convention: a value equal to a threshold earns
    the lower point count (``>`` comparisons in the regulation).
    """
    if value is None or value != value:  # None or NaN
        raise ValueError("missing nutrient value: the Nutri-Score cannot be computed")
    return bisect.bisect_left(list(thresholds), float(value))


def fvl_points(value: float, spec: Mapping) -> int:
    """Fruits/vegetables/legumes points (non-linear scale: 40 -> 1, 60 -> 2, 80 -> 5)."""
    if value is None or value != value:
        raise ValueError("missing fvl_pct: the Nutri-Score cannot be computed")
    idx = bisect.bisect_left(list(spec["thresholds"]), float(value))
    if idx == 0:
        return 0
    return int(list(spec["points"])[idx - 1])


def compute(
    nutrients: Mapping[str, float],
    version: str = "2023",
    cheese: bool = False,
    red_meat: bool = False,
    fibre_variant: str = "eu",
) -> dict:
    """Full breakdown of the Nutri-Score for one product.

    Parameters
    ----------
    nutrients :
        Mapping with the seven keys ``FEATURES`` (per 100 g).
    version :
        ``"2023"`` (current) or ``"2021"`` (original).
    cheese :
        Cheese always counts protein, even when N >= 11.
    red_meat :
        2023 only: protein points are capped at 2 for red meat products.
    fibre_variant :
        2021 only: ``"eu"`` (OFF / EU table, default) or ``"aoac"``.

    Returns
    -------
    dict with ``N``, ``p_fibre``, ``p_protein``, ``p_fvl``, ``P``, ``score``,
    ``grade`` and the ``protein_counted`` flag (useful for the report).
    """
    _check_version(version)
    missing = [k for k in FEATURES if k not in nutrients]
    if missing:
        raise KeyError(f"missing nutrient keys: {missing}")
    if version == "2021" and fibre_variant not in ("eu", "aoac"):
        raise ValueError("fibre_variant must be 'eu' or 'aoac'")

    T = load_thresholds()[version]

    # --- negative component N -------------------------------------------------
    N = points(nutrients["energy_kj"], T["energy_kj"])
    N += points(nutrients["sat_fat"], T["saturated_fat_g"])
    N += points(nutrients["sugars"], T["sugars_g"])
    if version == "2023":
        N += points(nutrients["salt"], T["salt_g"])
    else:
        sodium_mg = float(nutrients["salt"]) * float(T["salt_to_sodium_mg"])
        N += points(sodium_mg, T["sodium_mg"])

    # --- positive component P -------------------------------------------------
    fibre_table = T["fibre_g_aoac"] if (version == "2021" and fibre_variant == "aoac") else T["fibre_g"]
    p_fibre = points(nutrients["fibre"], fibre_table)
    p_protein = points(nutrients["proteins"], T["protein_g"])
    if red_meat and version == "2023":
        p_protein = min(p_protein, int(T["rules"]["red_meat_protein_cap"]))
    p_fvl = fvl_points(nutrients["fvl_pct"], T["fvl_pct"])

    cutoff = int(T["rules"]["protein_cutoff_n"])
    if N < cutoff or cheese:
        protein_counted = True
    elif version == "2021" and T["rules"].get("fvl5_protein_exemption") and p_fvl == 5:
        protein_counted = True
    else:
        protein_counted = False

    P = p_fibre + p_fvl + (p_protein if protein_counted else 0)
    s = N - P

    return {
        "version": version,
        "N": N,
        "p_fibre": p_fibre,
        "p_protein": p_protein,
        "p_fvl": p_fvl,
        "P": P,
        "protein_counted": protein_counted,
        "score": int(s),
        "grade": grade(s, version),
    }


def score(
    nutrients: Mapping[str, float],
    version: str = "2023",
    cheese: bool = False,
    red_meat: bool = False,
    fibre_variant: str = "eu",
) -> int:
    """Numeric score only (see :func:`compute` for the breakdown)."""
    return compute(
        nutrients, version=version, cheese=cheese, red_meat=red_meat, fibre_variant=fibre_variant
    )["score"]


def grade(value: float, version: str = "2023") -> str:
    """Letter grade for a numeric score (A < B < C < D < E)."""
    _check_version(version)
    g = load_thresholds()[version]["grades"]
    if value <= g["A_max"]:
        return "A"
    if value <= g["B_max"]:
        return "B"
    if value <= g["C_max"]:
        return "C"
    if value <= g["D_max"]:
        return "D"
    return "E"
