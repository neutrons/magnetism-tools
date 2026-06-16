"""Extract magnetic space group data from spglib 2.x.

spglib indexes MSGs by a UNI number (1–1651).  For each UNI number we
retrieve metadata and the abstract symmetry operations from the internal
database without needing a concrete crystal structure.

Notes
-----
- spglib 2.x uses attribute access (not dict) for MagneticSpaceGroupType.
- time_reversals from get_magnetic_symmetry_from_database uses 0 (unitary)
  and 1 (antiunitary); we convert to theta = +1 / -1.
- bns_symbol and og_symbol are not exposed by spglib 2.x; only the numbers.
"""

import spglib

N_MSG = 1651


def get_metadata(uni_number: int) -> dict:
    """Return labels and classification for a UNI number."""
    info = spglib.get_magnetic_spacegroup_type(uni_number)
    return {
        "uni_number": int(info.uni_number),
        "bns_number": info.bns_number,
        "og_number": info.og_number,
        "type": int(info.type),
        "parent_sg": int(info.number),
    }


def get_operators(uni_number: int) -> list[dict]:
    """Return general-position operators (W, t, theta) for a UNI number.

    Parameters
    ----------
    uni_number : int
        UNI sequential index (1–1651).

    Returns
    -------
    list of dict with keys:
        W     : 3x3 list of int – rotation matrix (row-major)
        t     : list of 3 float – fractional translation
        theta : int (+1 or -1) – time-reversal factor
    """
    ops = spglib.get_magnetic_symmetry_from_database(uni_number)

    result = []
    for W, t, tr in zip(
        ops["rotations"], ops["translations"], ops["time_reversals"]
    ):
        result.append(
            {
                "W": W.tolist(),
                "t": [round(float(x), 8) for x in t],
                "theta": 1 if int(tr) == 0 else -1,
            }
        )
    return result


def iter_all_msgs():
    """Yield (metadata dict, operators list) for every MSG (UNI 1–1651)."""
    for uni in range(1, N_MSG + 1):
        yield get_metadata(uni), get_operators(uni)
