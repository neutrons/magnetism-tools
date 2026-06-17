"""Magnetic form factors <j0> and <j2> for transition-metal and rare-earth ions.

Parameterization (P. J. Brown, International Tables for Crystallography
Vol. C, 4th ed., Table 4.4.5):

    <j0(s)> = A exp(-a s²) + B exp(-b s²) + C exp(-c s²) + D

    <j2(s)> = [A exp(-a s²) + B exp(-b s²) + C exp(-c s²) + D] · s²

where  s = sin θ / λ = |Q| / (4π)  with |Q| in Å⁻¹.

The s² prefactor for <j2> follows from the l=2 spherical Bessel function limit
j2(x) → x² / 15 as x → 0, ensuring <j2(0)> = 0.

For spin-only moments the form factor is simply f(Q) = <j0(Q)>.
When an orbital moment L is present the correction is:

    f(Q) = <j0> + (2/g - 1) <j2>

where g is the Landé g-factor and the factor (2/g − 1) = L / (L + 2S).
"""

from __future__ import annotations

import csv
import pathlib
import numpy as np

_DATA_DIR = pathlib.Path(__file__).parents[2] / "data"
_J0_PATH = _DATA_DIR / "j0.csv"
_J2_PATH = _DATA_DIR / "j2.csv"

_J0: dict[str, tuple] | None = None
_J2: dict[str, tuple] | None = None


def _load(path: pathlib.Path) -> dict[str, tuple]:
    table: dict[str, tuple] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ion = row["Ion"].strip()
            if not ion:
                continue
            table[ion] = tuple(
                float(row[k]) for k in ("A", "a", "B", "b", "C", "c", "D")
            )
    return table


def _get_j0() -> dict[str, tuple]:
    global _J0
    if _J0 is None:
        _J0 = _load(_J0_PATH)
    return _J0


def _get_j2() -> dict[str, tuple]:
    global _J2
    if _J2 is None:
        _J2 = _load(_J2_PATH)
    return _J2


def _eval(coeffs: tuple, Q_mag: float) -> float:
    A, a, B, b, C, c, D = coeffs
    s2 = (Q_mag / (4 * np.pi)) ** 2
    return A * np.exp(-a * s2) + B * np.exp(-b * s2) + C * np.exp(-c * s2) + D


def _resolve(ion: str, table: dict[str, tuple]) -> str:
    if ion in table:
        return ion
    lower = ion.lower()
    matches = [k for k in table if k.lower() == lower]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(
            f"Ambiguous ion {ion!r}; matches {matches}. "
            "Use the exact case-sensitive key."
        )
    raise KeyError(
        f"Ion {ion!r} not found. Call available_ions() to list valid labels."
    )


def j0(ion: str, Q_mag: float) -> float:
    """Return <j0(|Q|)> for *ion* at scattering vector magnitude *Q_mag* (Å⁻¹)."""
    key = _resolve(ion, _get_j0())
    return float(_eval(_get_j0()[key], Q_mag))


def j2(ion: str, Q_mag: float) -> float:
    """Return <j2(|Q|)> for *ion* at scattering vector magnitude *Q_mag* (Å⁻¹).

    Uses the parameterization <j2(s)> = [A exp(-as²) + … + D] · s²
    where s = |Q|/(4π), which correctly gives <j2(0)> = 0.
    """
    key = _resolve(ion, _get_j2())
    s2 = (Q_mag / (4 * np.pi)) ** 2
    return float(_eval(_get_j2()[key], Q_mag) * s2)


def form_factor(ion: str, Q_mag: float, *, c2: float = 0.0) -> float:
    """Return the magnetic form factor f(|Q|) = <j0> + c2 * <j2>.

    Parameters
    ----------
    ion : str
        Ion label, e.g. ``"Mn2+"``, ``"Cr3+"``, ``"Ru1+"``.
    Q_mag : float
        |Q| in Å⁻¹.
    c2 : float
        Coefficient of the <j2> correction.  For spin-only moments ``c2=0``
        (default).  For general moments ``c2 = 2/g - 1 = L/(L+2S)`` where
        *g* is the Landé g-factor.

    Returns
    -------
    float
        Dimensionless form factor.
    """
    f = j0(ion, Q_mag)
    if c2 != 0.0:
        f += c2 * j2(ion, Q_mag)
    return f


def available_ions(table: str = "j0") -> list[str]:
    """Return ion labels available in the given table (``"j0"`` or ``"j2"``)."""
    src = _get_j0() if table == "j0" else _get_j2()
    return sorted(src)
