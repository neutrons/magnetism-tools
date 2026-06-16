"""Query the MSG operator table for magnetic symmetry analysis.

Workflow
--------
Given a parent space group number, a commensurate propagation vector k, and
fractional coordinates of magnetic ion sites, this module:

1. Loads all MSGs whose parent SG matches.
2. Filters to those compatible with k  (little-group condition).
3. For each compatible MSG and each input site, returns:
   - the orbit of the site (symmetry-equivalent positions)
   - the basis vectors for symmetry-allowed magnetic moments

Moment transformation rule
--------------------------
Under operator ``(W, t, theta)``, a magnetic moment (axial vector) transforms
as ``m' = theta * det(W) * W @ m``.  A moment is site-symmetry-allowed if it
is invariant under all operators that fix the site modulo lattice translations.

k-compatibility condition
-------------------------
Operator ``(W, t, theta)`` is compatible with propagation vector k if
``W @ k ≡ theta * k  (mod reciprocal lattice vectors)``, i.e. W maps k to
either +k (unitary) or -k (antiunitary), up to a reciprocal lattice vector.
For k = 0 all operators satisfy this.
"""

from __future__ import annotations

import importlib.resources
import json
import pathlib
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_TABLE: list[dict] | None = None
_TABLE_PATH = pathlib.Path(__file__).parents[2] / "data" / "msg_operators.json"


def _load_table() -> list[dict]:
    global _TABLE
    if _TABLE is None:
        with _TABLE_PATH.open() as f:
            _TABLE = json.load(f)["groups"]
    return _TABLE


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------


def _is_integer_mod_centering(
    v: np.ndarray, centering: str, tol: float = 1e-4
) -> bool:
    """Return True if v is a reciprocal-lattice vector for the given centering.

    Centering conditions (on h,k,l components of v):
        P : all integers
        I : h+k+l even
        F : h+k, h+l, k+l all even
        A : k+l even
        B : h+l even
        C : h+k even
        R : -h+k+l divisible by 3 (obverse)
    """
    v_round = np.round(v)
    if not np.all(np.abs(v - v_round) < tol):
        return False
    h, k, l = v_round.astype(int)
    if centering == "P":
        return True
    if centering == "I":
        return (h + k + l) % 2 == 0
    if centering == "F":
        return (h + k) % 2 == 0 and (h + l) % 2 == 0 and (k + l) % 2 == 0
    if centering == "A":
        return (k + l) % 2 == 0
    if centering == "B":
        return (h + l) % 2 == 0
    if centering == "C":
        return (h + k) % 2 == 0
    if centering in ("R", "R(obv)", "Robv"):
        return (-h + k + l) % 3 == 0
    if centering in ("R(rev)", "Rrev"):
        return (h - k + l) % 3 == 0
    return True  # unknown centering: fall back to primitive check


def _is_k_compatible(
    W: np.ndarray,
    theta: int,
    k: np.ndarray,
    centering: str = "P",
    tol: float = 1e-4,
) -> bool:
    """Return True if (W, theta) maps k to ±k mod the reciprocal lattice.

    Note: for centered lattices (F, I, A, B, C, R) the reciprocal lattice
    is not the simple integer lattice — pass the correct ``centering`` to
    avoid spurious matches.
    """
    Wk = W @ k
    diff_plus = Wk - theta * k
    diff_minus = Wk + theta * k
    return _is_integer_mod_centering(
        diff_plus, centering, tol
    ) or _is_integer_mod_centering(diff_minus, centering, tol)


def _fixes_site(
    W: np.ndarray, t: np.ndarray, x: np.ndarray, tol: float = 1e-4
) -> bool:
    """Return True if (W, t) maps x to x modulo a lattice translation."""
    diff = W @ x + t - x
    return np.all(np.abs(diff - np.round(diff)) < tol)


def _moment_basis(site_ops: list[dict]) -> np.ndarray:
    """Return basis vectors for moments invariant under all site-symmetry ops.

    Solves the linear system (theta * det(W) * W - I) m = 0 simultaneously
    for all operators that fix the site.

    Returns
    -------
    basis : ndarray, shape (3, n_free)
        Columns are orthonormal basis vectors spanning the allowed subspace.
        n_free = 0 means no moment is allowed; n_free = 3 means fully free.
    """
    # Build the null space of the stacked constraint matrix
    rows = []
    for op in site_ops:
        W = np.array(op["W"], dtype=float)
        theta = op["theta"]
        det_W = np.linalg.det(W)
        # m' - m = 0  →  (theta * det(W) * W - I) m = 0
        rows.append(theta * det_W * W - np.eye(3))

    A = np.vstack(rows)

    _, s, Vt = np.linalg.svd(A)
    # Guard against zero matrix (s[0] == 0): use max(s[0], 1) so tol stays > 0
    s_max = s[0] if s.size else 0.0
    tol = max(A.shape) * np.finfo(float).eps * max(s_max, 1.0)

    # Vt rows corresponding to small singular values span the null space
    # SVD of (m, n) matrix: s has min(m,n) values; null space is last rows of Vt
    n_null = int(np.sum(s < tol)) if s.size else 3
    basis = Vt[-n_null:].T if n_null > 0 else np.zeros((3, 0))
    return basis


def _primitive_anti_translation(
    ops: list[dict], tol: float = 1e-4
) -> np.ndarray | None:
    """Return the primitive anti-translation of a type-IV MSG, or None.

    Scans operators for (W=I, theta=-1, t≠0).  In centered cells the same
    physical anti-translation τ appears in multiple centering cosets; the
    copy with the smallest fractional-coordinate magnitude is the canonical
    primitive choice.

    Only call this for type-IV MSGs — grey groups (type-II) also carry
    (W=I, t=centering-vector, -1) operators that are NOT genuine magnetic
    anti-translations.
    """
    I3 = np.eye(3)
    candidates: list[np.ndarray] = []
    for op in ops:
        t = np.array(op["t"], dtype=float)
        if (
            op["theta"] == -1
            and np.allclose(np.array(op["W"]), I3, atol=tol)
            and not np.allclose(t % 1.0, np.zeros(3), atol=tol)
        ):
            candidates.append(t)
    if not candidates:
        return None
    return min(candidates, key=lambda v: float(np.linalg.norm(v % 1.0)))


def _orbit(
    ops: list[dict], x: np.ndarray, tol: float = 1e-4
) -> list[np.ndarray]:
    """Return the orbit of site x under all operators, reduced to [0,1)."""
    seen: list[np.ndarray] = []
    for op in ops:
        W = np.array(op["W"], dtype=float)
        t = np.array(op["t"], dtype=float)
        xp = (W @ x + t) % 1.0
        if not any(np.allclose(xp, s, atol=tol) for s in seen):
            seen.append(xp)
    return seen


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SiteResult:
    site: np.ndarray  # input fractional coordinates
    orbit: list[np.ndarray]  # all symmetry-equivalent positions
    moment_basis: np.ndarray  # columns = allowed moment directions (3 × n)
    n_free: int  # number of free moment components

    @property
    def is_magnetic(self) -> bool:
        return self.n_free > 0


@dataclass
class MSGResult:
    uni_number: int
    bns_number: str
    og_number: str
    msg_type: int
    parent_sg: int
    n_ops: int
    # Fractional coordinates of the MSG origin in the parent cell.  Two results
    # with the same BNS number but different origin_shift represent physically
    # distinct magnetic orderings (related by the anti-translation — time-reversed
    # domain states).  The default (0,0,0) corresponds to the standard setting.
    origin_shift: tuple = (0.0, 0.0, 0.0)
    sites: list[SiteResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compatible_msgs(
    parent_sg: int,
    k: array_like,
    sites: list[array_like],
    *,
    centering: str = "P",
    k_tol: float = 1e-4,
    site_tol: float = 1e-4,
) -> list[MSGResult]:
    """Find MSGs compatible with the given parent SG and k-vector.

    Parameters
    ----------
    parent_sg : int
        ITA number of the parent (non-magnetic) space group (1–230).
    k : array_like, shape (3,)
        Commensurate propagation vector in fractional reciprocal coordinates.
    sites : list of array_like, shape (3,)
        Fractional coordinates of the magnetic ion sites.
    centering : str
        Lattice centering of the parent cell: "P", "I", "F", "A", "B", "C",
        "R".  Determines which difference vectors W@k - theta*k are considered
        reciprocal-lattice vectors.  Defaults to "P" (primitive).  For
        face-centred parents (e.g. Fm-3m, SG 225) pass centering="F".
    k_tol : float
        Tolerance for the k-compatibility check.
    site_tol : float
        Tolerance for orbit and site-symmetry comparisons.

    Returns
    -------
    list of MSGResult, one per compatible MSG, sorted by uni_number.
    """
    k = np.asarray(k, dtype=float)
    sites = [np.asarray(s, dtype=float) for s in sites]

    results = []
    for entry in _load_table():
        if entry["parent_sg"] != parent_sg:
            continue

        ops = entry["operators"]

        # Filter operators to those compatible with k
        k_ops = [
            op
            for op in ops
            if _is_k_compatible(
                np.array(op["W"]),
                op["theta"],
                k,
                centering=centering,
                tol=k_tol,
            )
        ]

        if not k_ops:
            continue

        # For type-IV MSGs, the primitive anti-translation τ defines two inequivalent
        # origin choices: p=0 and p=τ.  These produce physically distinct orderings
        # (time-reversed domain states) and are listed separately, matching the
        # convention of Bilbao's MAXMAGN [1]_.
        # Only type-IV MSGs carry a genuine magnetic anti-translation — type-II
        # (grey) operators with (W=I, t=centering-vector, -1) are excluded here
        # by checking the MSG type from the database entry.
        origin_shifts: list[np.ndarray] = [np.zeros(3)]
        if entry["type"] == 4:
            tau = _primitive_anti_translation(k_ops, tol=site_tol)
            if tau is not None:
                origin_shifts.append(tau)

        for p in origin_shifts:
            # Build per-site results: shift each site into the MSG frame (x - p),
            # compute site-symmetry and orbit there, then convert orbit back to the
            # parent frame (+ p mod 1).
            site_results = []
            for x in sites:
                x_p = x - p  # site in MSG frame

                site_ops = [
                    op
                    for op in k_ops
                    if _fixes_site(
                        np.array(op["W"]), np.array(op["t"]), x_p, site_tol
                    )
                ]

                basis = _moment_basis(site_ops)
                orbit_msg = _orbit(k_ops, x_p, site_tol)
                orbit = [(pos + p) % 1.0 for pos in orbit_msg]

                site_results.append(
                    SiteResult(
                        site=x,
                        orbit=orbit,
                        moment_basis=basis,
                        n_free=basis.shape[1],
                    )
                )

            results.append(
                MSGResult(
                    uni_number=entry["uni_number"],
                    bns_number=entry["bns_number"],
                    og_number=entry["og_number"],
                    msg_type=entry["type"],
                    parent_sg=entry["parent_sg"],
                    n_ops=len(ops),
                    origin_shift=tuple(float(v) for v in p.round(6)),
                    sites=site_results,
                )
            )

    # Sort by (uni_number, origin_shift) for deterministic ordering
    results.sort(key=lambda r: (r.uni_number, r.origin_shift))
    return results


def subgroup_msgs(
    parent_sg: int,
    k: array_like,
    sites: list[array_like],
    *,
    centering: str = "P",
    k_tol: float = 1e-4,
    site_tol: float = 1e-4,
) -> list[MSGResult]:
    """Find all MSGs compatible with k that are geometric subgroups of the parent SG.

    Inspired by the Bilbao k-SUBGROUPSMAG workflow [1]_ — searches across all parent SGs
    down to MSG 1.1, not just the input parent SG.

    Algorithm
    ---------
    1. Compute the little co-group of k: the set of unique rotation matrices W
       from the parent SG's type-I MSG that map k to ±k (mod the centering
       reciprocal lattice).
    2. Search all 1651 MSGs for those whose W matrices are all contained in
       that set.  This is the geometric subgroup criterion in the given basis.
    3. For each candidate MSG, use all its operators for site-symmetry analysis
       (k-compatibility is guaranteed by the little co-group filter).

    Parameters
    ----------
    parent_sg : int
        ITA number of the parent space group (1–230).
    k : array_like, shape (3,)
        Commensurate propagation vector in fractional reciprocal coordinates.
    sites : list of array_like, shape (3,)
        Fractional coordinates of the magnetic ion sites.
    centering : str
        Lattice centering of the parent cell (``"P"``, ``"R"``, ``"F"``, …).
    k_tol, site_tol : float
        Tolerances for k-compatibility and site-fixing comparisons.

    Returns
    -------
    list of MSGResult sorted by n_ops descending (highest-symmetry first).
    Includes all MSG types (I–IV) and all parent SGs in the subgroup lattice.

    References
    ----------
    .. [1] J. M. Perez-Mato et al., *Annu. Rev. Mater. Res.* **45**, 217 (2015).
    """
    k = np.asarray(k, dtype=float)
    sites_np = [np.asarray(s, dtype=float) for s in sites]
    table = _load_table()

    # Step 1: build the little co-group W set from the parent SG type-I MSG.
    # Include W if k-compatible for theta=+1 OR theta=-1 (covers antiunitary ops
    # in subgroup MSGs).
    parent_type1 = next(
        (e for e in table if e["parent_sg"] == parent_sg and e["type"] == 1),
        None,
    )
    if parent_type1 is None:
        return []

    lcg_W: list[np.ndarray] = []
    for op in parent_type1["operators"]:
        W = np.array(op["W"])
        compat_plus = _is_k_compatible(W, +1, k, centering, k_tol)
        compat_minus = _is_k_compatible(W, -1, k, centering, k_tol)
        if (compat_plus or compat_minus) and not any(
            np.allclose(W, Wm, atol=k_tol) for Wm in lcg_W
        ):
            lcg_W.append(W)

    if not lcg_W:
        return []

    # Step 2: find all MSGs whose W matrices are a subset of lcg_W.
    results = []
    for entry in table:
        ops = entry["operators"]

        # Collect unique W matrices for this MSG
        msg_Ws: list[np.ndarray] = []
        for op in ops:
            W = np.array(op["W"])
            if not any(np.allclose(W, Wm, atol=k_tol) for Wm in msg_Ws):
                msg_Ws.append(W)

        # All W matrices must appear in the little co-group
        if not all(
            any(np.allclose(W, Wm, atol=k_tol) for Wm in lcg_W) for W in msg_Ws
        ):
            continue

        # Step 3: site-symmetry analysis using ALL operators of the MSG.
        site_results = []
        for x in sites_np:
            site_ops = [
                op
                for op in ops
                if _fixes_site(
                    np.array(op["W"]), np.array(op["t"]), x, site_tol
                )
            ]
            basis = _moment_basis(site_ops)
            orbit = _orbit(ops, x, site_tol)
            site_results.append(
                SiteResult(
                    site=x,
                    orbit=orbit,
                    moment_basis=basis,
                    n_free=basis.shape[1],
                )
            )

        results.append(
            MSGResult(
                uni_number=entry["uni_number"],
                bns_number=entry["bns_number"],
                og_number=entry["og_number"],
                msg_type=entry["type"],
                parent_sg=entry["parent_sg"],
                n_ops=len(ops),
                sites=site_results,
            )
        )

    # Sort by n_ops descending (highest symmetry first), break ties by uni_number
    results.sort(key=lambda r: (-r.n_ops, r.uni_number))
    return results


def maximal_msgs(
    parent_sg: int,
    k: array_like,
    sites: list[array_like],
    **kwargs,
) -> list[MSGResult]:
    """Return k-maximal compatible MSGs that allow a non-zero moment.

    Implements the Bilbao MAXMAGN definition [1]_: a MSG is k-maximal if no
    k-compatible supergroup exists in the subgroup lattice of G1'.

    For **k = 0** the little co-group equals the full parent SG point group, so
    ``subgroup_msgs`` would include MSGs from unrelated SGs that happen to share
    the same point group.  ``compatible_msgs`` (which restricts to the input
    parent SG) is used instead; it already enumerates both origin shifts for
    type-IV MSGs.

    For **k ≠ 0** the little co-group is a strict subgroup of the parent SG
    point group.  ``subgroup_msgs`` is used so the true k-maximal MSG is found
    even when it has a *different* parent SG than the input — e.g. α-RuCl₃
    (parent = 148) whose maximal MSG is BNS 2.7 (parent = 2).  For each
    type-IV MSG at the maximum symmetry level, a second result with the
    primitive anti-translation origin shift τ is appended.

    Parameters
    ----------
    parent_sg : int
        ITA number of the paramagnetic parent space group.
    k : array_like, shape (3,)
        Commensurate propagation vector in fractional reciprocal coordinates.
    sites : list of array_like, shape (3,)
        Fractional coordinates of the magnetic ion sites.
    **kwargs
        Passed to ``compatible_msgs`` / ``subgroup_msgs``: ``centering``,
        ``k_tol``, ``site_tol``.

    Returns
    -------
    list of MSGResult, sorted by (uni_number, origin_shift).

    References
    ----------
    .. [1] J. M. Perez-Mato et al., *Annu. Rev. Mater. Res.* **45**, 217 (2015).
    """
    k_np = np.asarray(k, dtype=float)
    k_tol = kwargs.get("k_tol", 1e-4)
    site_tol = kwargs.get("site_tol", 1e-4)
    centering = kwargs.get("centering", "P")
    sites_np = [np.asarray(s, dtype=float) for s in sites]

    is_zone_center = np.allclose(k_np, 0, atol=k_tol)

    if is_zone_center:
        # k = 0: search within the parent SG family.
        # compatible_msgs already handles origin-shift enumeration for type-IV MSGs.
        base = compatible_msgs(parent_sg, k_np, sites, **kwargs)
        magnetic = [r for r in base if any(s.n_free > 0 for s in r.sites)]
        if not magnetic:
            return []
        max_n = max(r.n_ops for r in magnetic)
        return [r for r in magnetic if r.n_ops == max_n]

    # k ≠ 0: search the full subgroup lattice.
    # subgroup_msgs guarantees all operators are k-compatible (W-matrix subset
    # check), so n_ops comparisons correctly reflect true symmetry level.
    base = subgroup_msgs(parent_sg, k_np, sites, **kwargs)
    magnetic = [r for r in base if any(s.n_free > 0 for s in r.sites)]
    if not magnetic:
        return []

    max_n = max(r.n_ops for r in magnetic)
    top = [r for r in magnetic if r.n_ops == max_n]

    # Enumerate the τ-shifted origin for type-IV MSGs at the maximum level.
    # subgroup_msgs returns results with origin_shift = (0,0,0); add the τ variant.
    final: list[MSGResult] = []
    for r in top:
        final.append(r)
        if r.msg_type != 4:
            continue
        entry = next(
            e for e in _load_table() if e["bns_number"] == r.bns_number
        )
        ops = entry["operators"]
        k_ops = [
            op
            for op in ops
            if _is_k_compatible(
                np.array(op["W"]),
                op["theta"],
                k_np,
                centering=centering,
                tol=k_tol,
            )
        ]
        tau = _primitive_anti_translation(k_ops, tol=site_tol)
        if tau is None or np.allclose(tau, 0, atol=site_tol):
            continue
        site_results = []
        for x in sites_np:
            x_p = x - tau
            site_ops = [
                op
                for op in k_ops
                if _fixes_site(
                    np.array(op["W"]), np.array(op["t"]), x_p, site_tol
                )
            ]
            basis = _moment_basis(site_ops)
            orbit_msg = _orbit(k_ops, x_p, site_tol)
            orbit = [(pos + tau) % 1.0 for pos in orbit_msg]
            site_results.append(
                SiteResult(
                    site=x,
                    orbit=orbit,
                    moment_basis=basis,
                    n_free=basis.shape[1],
                )
            )
        final.append(
            MSGResult(
                uni_number=entry["uni_number"],
                bns_number=entry["bns_number"],
                og_number=entry["og_number"],
                msg_type=entry["type"],
                parent_sg=entry["parent_sg"],
                n_ops=len(ops),
                origin_shift=tuple(float(v) for v in tau.round(6)),
                sites=site_results,
            )
        )

    final.sort(key=lambda r: (r.uni_number, r.origin_shift))
    return final


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def find_by_bns(bns_number: str) -> MSGResult | None:
    """Return an MSGResult (no site data) for the MSG with the given BNS number.

    Parameters
    ----------
    bns_number : str
        BNS label as a string, e.g. ``"148.17"``, ``"2.7"``.  Leading/trailing
        whitespace is stripped.

    Returns
    -------
    MSGResult with an empty ``sites`` list, or ``None`` if not found.
    """
    bns_number = bns_number.strip()
    for entry in _load_table():
        if entry["bns_number"] == bns_number:
            return MSGResult(
                uni_number=entry["uni_number"],
                bns_number=entry["bns_number"],
                og_number=entry["og_number"],
                msg_type=entry["type"],
                parent_sg=entry["parent_sg"],
                n_ops=len(entry["operators"]),
                sites=[],
            )
    return None


def analyze_msg(
    bns_number: str,
    sites: list[array_like],
    *,
    site_tol: float = 1e-4,
) -> MSGResult | None:
    """Compute site-symmetry moment basis for a specific MSG identified by BNS number.

    This is the lower-symmetry entry point: the user has already identified
    the MSG (e.g. from Bilbao/Jana subgroup tables) and wants to know which
    moment directions are symmetry-allowed at each site.

    Unlike ``compatible_msgs``, this function skips the k-compatibility filter
    — all operators of the MSG are used as site-symmetry candidates.  Use this
    when the MSG already encodes the k-vector (e.g. a type-IV MSG whose
    anti-translation captures the magnetic doubling).

    Parameters
    ----------
    bns_number : str
        BNS label of the target MSG, e.g. ``"2.7"`` for P_S 1̄.
    sites : list of array_like, shape (3,)
        Fractional coordinates of the magnetic ion sites in the MSG's cell.
    site_tol : float
        Tolerance for site-fixing comparisons.

    Returns
    -------
    MSGResult with site data, or ``None`` if the BNS number is not found.
    """
    entry = next(
        (e for e in _load_table() if e["bns_number"] == bns_number.strip()),
        None,
    )
    if entry is None:
        return None

    sites_np = [np.asarray(s, dtype=float) for s in sites]
    ops = entry["operators"]

    site_results = []
    for x in sites_np:
        site_ops = [
            op
            for op in ops
            if _fixes_site(np.array(op["W"]), np.array(op["t"]), x, site_tol)
        ]
        basis = _moment_basis(site_ops)
        orbit = _orbit(ops, x, site_tol)
        site_results.append(
            SiteResult(
                site=x,
                orbit=orbit,
                moment_basis=basis,
                n_free=basis.shape[1],
            )
        )

    return MSGResult(
        uni_number=entry["uni_number"],
        bns_number=entry["bns_number"],
        og_number=entry["og_number"],
        msg_type=entry["type"],
        parent_sg=entry["parent_sg"],
        n_ops=len(ops),
        sites=site_results,
    )


def domain_operators(
    msg_bns: str,
    parent_sg: int,
    tol: float = 1e-4,
) -> list[dict]:
    """Return spatial operators of the parent SG that are absent from the MSG.

    These are the "broken symmetry" operations — applying any one of them to a
    single-domain state generates a distinct but symmetry-equivalent domain.
    The number of domains equals ``len(domain_operators(...)) + 1``.

    This comparison is meaningful when the MSG is a subgroup of the parent SG
    in the *same cell* (type-I or type-III MSGs).  For type-IV MSGs whose cell
    differs from the parent (e.g. P_S 1̄ derived from R-3), the function still
    returns a useful approximation by comparing only the rotational parts of
    the operators (ignoring translations that differ by the cell doubling).

    Parameters
    ----------
    msg_bns : str
        BNS number of the MSG (e.g. ``"148.17"``).
    parent_sg : int
        ITA number of the parent space group (e.g. ``148`` for R-3).
    tol : float
        Tolerance for rotation-matrix comparison.

    Returns
    -------
    List of operator dicts from the parent SG type-I MSG whose rotation
    matrix does not appear (with any translation or theta) among the MSG
    operators.

    Notes
    -----
    The number of magnetic domains is::

        n_domains = n_unique_W_in_parent / n_unique_W_in_MSG

    where both counts use the *point-group* (unique rotation matrices,
    ignoring translations).  For CrCl3 (R-3 → P_S1̄, BNS 2.7):
    ``|S6| / |Ci| = 6 / 2 = 3`` domains, while ``len(domain_operators(...))``
    returns 12 (4 distinct broken rotations × 3 R-centering copies).

    For type-III MSGs within the same parent SG, this function returns an
    empty list because every rotation matrix of the parent is still present
    in the MSG (some as antiunitary operations).  Domains in that case arise
    from time-reversal symmetry breaking, not spatial symmetry lowering.
    """
    table = _load_table()

    # Get all operators of the parent SG type-I MSG (purely unitary)
    parent_type1 = next(
        (e for e in table if e["parent_sg"] == parent_sg and e["type"] == 1),
        None,
    )
    if parent_type1 is None:
        return []
    parent_ops = parent_type1["operators"]

    # Get rotation matrices present in the MSG (any theta, any t)
    msg_entry = next(
        (e for e in table if e["bns_number"] == msg_bns.strip()), None
    )
    if msg_entry is None:
        return []
    msg_rotations = [np.array(op["W"]) for op in msg_entry["operators"]]

    broken = []
    for op in parent_ops:
        W = np.array(op["W"])
        if not any(np.allclose(W, Wm, atol=tol) for Wm in msg_rotations):
            broken.append(op)

    return broken


# type alias for cleaner annotations above
array_like = "np.ndarray | list"
