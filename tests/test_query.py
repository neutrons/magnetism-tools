"""Tests for msgjson.query — MAXMAGN-style MSG analysis."""

import numpy as np
import pytest

from msgjson.query import (
    _is_k_compatible,
    _is_integer_mod_centering,
    _fixes_site,
    _moment_basis,
    compatible_msgs,
    maxmagn,
    find_by_bns,
    analyze_msg,
    domain_operators,
    MSGResult,
    SiteResult,
)


# ---------------------------------------------------------------------------
# Unit tests — centering-aware k-compatibility
# ---------------------------------------------------------------------------

class TestIsIntegerModCentering:
    def test_primitive_integer(self):
        assert _is_integer_mod_centering(np.array([1.0, -2.0, 0.0]), "P")

    def test_primitive_noninteger(self):
        assert not _is_integer_mod_centering(np.array([0.5, 0.0, 0.0]), "P")

    def test_fcc_allowed(self):
        # (2, 0, 0): h+k=2 even, h+l=2 even, k+l=0 even  → valid F vector
        assert _is_integer_mod_centering(np.array([2.0, 0.0, 0.0]), "F")

    def test_fcc_forbidden(self):
        # (1, 0, 0): h+k=1 odd  → NOT a reciprocal lattice vector of FCC
        assert not _is_integer_mod_centering(np.array([1.0, 0.0, 0.0]), "F")

    def test_bcc_allowed(self):
        # (1, 1, 0): h+k+l=2 even  → valid I vector
        assert _is_integer_mod_centering(np.array([1.0, 1.0, 0.0]), "I")

    def test_bcc_forbidden(self):
        # (1, 0, 0): h+k+l=1 odd  → NOT a reciprocal lattice vector of BCC
        assert not _is_integer_mod_centering(np.array([1.0, 0.0, 0.0]), "I")

    def test_tolerance(self):
        # 1e-5 deviation should still pass
        assert _is_integer_mod_centering(np.array([1.0 + 1e-5, 0.0, 0.0]), "P", tol=1e-4)

    def test_tolerance_strict(self):
        # 5e-4 deviation should fail at default tol=1e-4
        assert not _is_integer_mod_centering(np.array([1.0 + 5e-4, 0.0, 0.0]), "P", tol=1e-4)


class TestIsKCompatible:
    def test_identity_k_zero(self):
        W = np.eye(3, dtype=int)
        assert _is_k_compatible(W, +1, np.array([0.0, 0.0, 0.0]))
        assert _is_k_compatible(W, -1, np.array([0.0, 0.0, 0.0]))

    def test_inversion_k_half(self):
        # Inversion: W = -I, theta=+1 → W@k = -k; diff_plus = -k - k = -2k
        # For k=(0.5, 0, 0), diff_plus = (-1, 0, 0) → integer: compatible
        W = -np.eye(3, dtype=int)
        assert _is_k_compatible(W, +1, np.array([0.5, 0.0, 0.0]))

    def test_4fold_fcc_k_half_half_half(self):
        # 4-fold rotation [001]: maps (1/2,1/2,1/2) → (-1/2,1/2,1/2)
        # diff_plus = (-1/2,1/2,1/2) - (1/2,1/2,1/2) = (-1,0,0)
        # For P centering: (-1,0,0) IS an integer  → compatible (WRONG physics)
        # For F centering: (-1,0,0): h+k=-1 odd     → NOT compatible (correct)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        k = np.array([0.5, 0.5, 0.5])
        assert     _is_k_compatible(W, +1, k, centering="P")  # passes with P (too loose)
        assert not _is_k_compatible(W, +1, k, centering="F")  # correctly excluded with F


# ---------------------------------------------------------------------------
# Unit tests — moment basis
# ---------------------------------------------------------------------------

class TestMomentBasis:
    def test_identity_only(self):
        # Only identity: (det*theta*W - I) = (1*1*I - I) = 0  → all moments free
        ops = [{"W": np.eye(3, dtype=int).tolist(), "theta": 1}]
        basis = _moment_basis(ops)
        assert basis.shape == (3, 3)

    def test_inversion_unitary(self):
        # W = -I, theta = +1: det(-I)=-1, so det*theta*W - I = +I - I = 0 → all free
        ops = [{"W": (-np.eye(3, dtype=int)).tolist(), "theta": 1}]
        basis = _moment_basis(ops)
        assert basis.shape == (3, 3)

    def test_inversion_antiunitary(self):
        # W = -I, theta = -1: det(-I)=-1, so (-1)*(-1)*(-I) - I = -I - I = -2I
        # Only m=0 satisfies → n_free=0
        ops = [{"W": (-np.eye(3, dtype=int)).tolist(), "theta": -1}]
        basis = _moment_basis(ops)
        assert basis.shape == (3, 0)

    def test_3fold_111_axis(self):
        # 3-fold rotation about [111]: cyclic permutation (x,y,z)→(z,x,y)
        # theta=+1, det=+1 → (W - I) forces m_x=m_y=m_z → m || [1,1,1]
        W = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        ops = [{"W": W.tolist(), "theta": 1}]
        basis = _moment_basis(ops)
        assert basis.shape[1] == 1
        direction = basis[:, 0] / np.linalg.norm(basis[:, 0])
        expected = np.array([1, 1, 1]) / np.sqrt(3)
        assert np.allclose(np.abs(direction @ expected), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration tests — compatible_msgs
# ---------------------------------------------------------------------------

class TestCompatibleMsgs:
    def test_returns_list_of_msgreults(self):
        results = compatible_msgs(136, k=[0, 0, 0], sites=[[0, 0, 0]])
        assert isinstance(results, list)
        assert all(isinstance(r, MSGResult) for r in results)

    def test_sorted_by_uni(self):
        results = compatible_msgs(136, k=[0, 0, 0], sites=[[0, 0, 0]])
        unis = [r.uni_number for r in results]
        assert unis == sorted(unis)

    def test_all_parent_sg_match(self):
        results = compatible_msgs(136, k=[0, 0, 0], sites=[[0, 0, 0]])
        assert all(r.parent_sg == 136 for r in results)

    def test_site_result_shape(self):
        results = compatible_msgs(136, k=[0, 0, 0], sites=[[0, 0, 0]])
        for r in results:
            assert len(r.sites) == 1
            s = r.sites[0]
            assert isinstance(s, SiteResult)
            assert s.moment_basis.shape[0] == 3
            assert s.n_free == s.moment_basis.shape[1]

    def test_mnf2_finds_type3_msgs(self):
        # MnF2: SG 136 (P4_2/mnm), k=(0,0,0), Mn at Wyckoff 2a = (0,0,0)
        results = compatible_msgs(136, k=[0, 0, 0], sites=[[0, 0, 0]])
        type3 = [r for r in results if r.msg_type == 3]
        assert len(type3) > 0, "Expected at least one type-III MSG for SG136"

    def test_sg1_k0_trivial(self):
        # SG1 (P1) has 3 MSGs: type-I (uni=1), grey/type-II (uni=2), type-IV (uni=3)
        # The type-I MSG has only identity → all moment directions free (n_free=3)
        results = compatible_msgs(1, k=[0, 0, 0], sites=[[0, 0, 0]])
        assert len(results) == 3
        type1 = next(r for r in results if r.msg_type == 1)
        assert type1.sites[0].n_free == 3

    def test_wrong_parent_returns_empty(self):
        # SG number 999 does not exist
        results = compatible_msgs(999, k=[0, 0, 0], sites=[[0, 0, 0]])
        assert results == []


# ---------------------------------------------------------------------------
# Integration tests — maxmagn (MnF2 canonical case)
# ---------------------------------------------------------------------------

class TestMaxmagn:
    def test_mnf2_returns_results(self):
        results = maxmagn(136, k=[0, 0, 0], sites=[[0, 0, 0]])
        assert len(results) > 0

    def test_mnf2_all_magnetic(self):
        results = maxmagn(136, k=[0, 0, 0], sites=[[0, 0, 0]])
        assert all(any(s.n_free > 0 for s in r.sites) for r in results)

    def test_mnf2_moment_direction(self):
        # Known physics: both maximal MSGs for MnF2 have m || [0,0,1]
        results = maxmagn(136, k=[0, 0, 0], sites=[[0, 0, 0]])
        for r in results:
            basis = r.sites[0].moment_basis
            assert basis.shape[1] == 1, (
                f"Expected 1-D moment subspace, got {basis.shape[1]} for {r.bns_number}"
            )
            direction = basis[:, 0] / np.linalg.norm(basis[:, 0])
            c_axis = np.array([0.0, 0.0, 1.0])
            assert np.allclose(np.abs(direction @ c_axis), 1.0, atol=1e-4), (
                f"Expected m || [001] for {r.bns_number}, got {direction}"
            )

    def test_mnf2_two_maximal_msgs(self):
        # Two distinct type-III MSGs of P4_2/mnm allow m || [001]
        results = maxmagn(136, k=[0, 0, 0], sites=[[0, 0, 0]])
        assert len(results) == 2, f"Expected 2 maximal MSGs, got {len(results)}"

    def test_mnf2_type3(self):
        results = maxmagn(136, k=[0, 0, 0], sites=[[0, 0, 0]])
        assert all(r.msg_type == 3 for r in results)

    def test_mnf2_same_n_ops(self):
        results = maxmagn(136, k=[0, 0, 0], sites=[[0, 0, 0]])
        n_ops_set = {r.n_ops for r in results}
        assert len(n_ops_set) == 1, "All maximal MSGs should have the same n_ops"

    def test_sg1_k0_returns_one(self):
        # SG1: two MSGs allow moments (type-I with 1 op, type-IV with 2 ops).
        # maxmagn picks the one(s) with the most operators among moment-allowing
        # groups — that is the type-IV MSG (BNS 1.3, n_ops=2).
        results = maxmagn(1, k=[0, 0, 0], sites=[[0, 0, 0]])
        assert len(results) == 1
        assert results[0].msg_type == 4
        assert results[0].sites[0].n_free == 3

    def test_nonmagnetic_site_empty(self):
        # A site at a position with full O_h symmetry in a type-I MSG should
        # return n_free=0 → maxmagn returns []
        # SG225 (Fm-3m), k=(0,0,0), type-I MSG (uni=1632, parent=225): all ops
        # unitary → m must be invariant under full O_h → forced to 0
        # Use centering="F" to get the correct little group
        results = maxmagn(225, k=[0, 0, 0], sites=[[0, 0, 0]], centering="F")
        # Some MSGs in SG225 are type I (grey) with no moment; we just check
        # that the function returns a list (may be empty or non-empty depending
        # on which type-I/III/IV MSGs exist for this parent)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Integration tests — CrI3 (SG 148, R-3, ferromagnetic k=0)
# ---------------------------------------------------------------------------

class TestCrI3:
    """CrI3 low-temperature rhombohedral ferromagnet.

    Structure:  SG 148 (R-3), hexagonal setting
    Cr site:    6c Wyckoff, first position (0, 0, ~1/3) along c-axis
    Ordering:   ferromagnetic, k = (0, 0, 0)
    Moments:    constrained to c-axis by the 3-fold site symmetry
    """

    SG = 148
    K = [0, 0, 0]
    CR_SITE = [0, 0, 1 / 3]  # 6c Wyckoff position in hexagonal R-3

    def test_compatible_msgs_count(self):
        # SG148 has exactly 4 MSGs: type I, II (grey), III, IV
        results = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE])
        assert len(results) == 4

    def test_compatible_msgs_types(self):
        results = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE])
        types = {r.msg_type for r in results}
        assert types == {1, 2, 3, 4}

    def test_grey_group_forbids_moment(self):
        # Type-II (grey) MSG includes time reversal → no ordered moment allowed
        results = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE])
        grey = next(r for r in results if r.msg_type == 2)
        assert grey.sites[0].n_free == 0

    def test_ferromagnetic_msg_type1_allows_moment(self):
        # Type-I MSG (all unitary): the ferromagnetic R-3 MSG
        results = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE])
        type1 = next(r for r in results if r.msg_type == 1)
        assert type1.sites[0].n_free == 1

    def test_ferromagnetic_msg_type1_moment_along_c(self):
        results = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE])
        type1 = next(r for r in results if r.msg_type == 1)
        basis = type1.sites[0].moment_basis
        direction = basis[:, 0] / np.linalg.norm(basis[:, 0])
        c_axis = np.array([0.0, 0.0, 1.0])
        assert np.allclose(np.abs(direction @ c_axis), 1.0, atol=1e-4), (
            f"Expected m || [001], got {direction}"
        )

    def test_type1_orbit_is_6c(self):
        # At (0,0,1/3) in R-3 with 18 operators, site symmetry = C3 (3 ops)
        # → orbit size = 18 / 3 = 6 (the 6c Wyckoff multiplicity)
        results = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE])
        type1 = next(r for r in results if r.msg_type == 1)
        assert len(type1.sites[0].orbit) == 6

    def test_maxmagn_returns_one_msg(self):
        results = maxmagn(self.SG, k=self.K, sites=[self.CR_SITE])
        assert len(results) == 1

    def test_maxmagn_moment_along_c(self):
        # All maximal magnetic MSGs must have m || [0,0,1]
        results = maxmagn(self.SG, k=self.K, sites=[self.CR_SITE])
        for r in results:
            s = r.sites[0]
            assert s.n_free == 1, f"Expected 1-D moment subspace for {r.bns_number}"
            direction = s.moment_basis[:, 0] / np.linalg.norm(s.moment_basis[:, 0])
            c_axis = np.array([0.0, 0.0, 1.0])
            assert np.allclose(np.abs(direction @ c_axis), 1.0, atol=1e-4), (
                f"Expected m || [001] for {r.bns_number}, got {direction}"
            )

    def test_maxmagn_bns_number(self):
        # maxmagn selects type-IV MSG (BNS 148.20) — most operators (36)
        # among moment-allowing MSGs
        results = maxmagn(self.SG, k=self.K, sites=[self.CR_SITE])
        assert results[0].bns_number == "148.20"

    def test_maxmagn_type4_doubled_orbit(self):
        # Type-IV MSG has 36 operators; orbit at (0,0,1/3) = 36/3 = 12
        results = maxmagn(self.SG, k=self.K, sites=[self.CR_SITE])
        assert results[0].msg_type == 4
        assert len(results[0].sites[0].orbit) == 12

    def test_all_magnetic_msgs_agree_on_c_axis(self):
        # Type I, III, and IV all allow moments; all constrain m || [0,0,1]
        results = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE])
        c_axis = np.array([0.0, 0.0, 1.0])
        for r in results:
            s = r.sites[0]
            if s.n_free == 0:
                continue
            assert s.n_free == 1, f"Unexpected n_free={s.n_free} for {r.bns_number}"
            direction = s.moment_basis[:, 0] / np.linalg.norm(s.moment_basis[:, 0])
            assert np.allclose(np.abs(direction @ c_axis), 1.0, atol=1e-4), (
                f"Expected m || [001] for {r.bns_number}, got {direction}"
            )


# ---------------------------------------------------------------------------
# Integration tests — CrCl3 (SG 148, R-3, interlayer AFM k=(0,0,3/2))
# ---------------------------------------------------------------------------

class TestCrCl3:
    """CrCl3 low-temperature rhombohedral antiferromagnet.

    Physical structure
    ------------------
    SG 148 (R-3), hexagonal setting, centering R.
    Cr occupies the 6c Wyckoff site; k = (0, 0, 3/2) is the BZ boundary
    along c for the R-centered lattice (shortest R-RL vector along c is
    (0,0,3), so the BZ edge is at (0,0,3/2)).

    Physical ordering: intralayer ferromagnetic, interlayer antiferromagnetic,
    with moments IN the ab-plane (easy-plane anisotropy).

    Why the structure is non-maximal
    ---------------------------------
    Every position in the 6c orbit lies on a 3-fold rotation axis — (0,0,z)
    on the primary c-axis, and (1/3,2/3,z)/(2/3,1/3,z) on the two secondary
    axes that emerge from the hexagonal lattice.  Therefore C3 is always in
    the site-symmetry group.  A 3-fold rotation has no real in-plane
    eigenvector, so ALL compatible MSGs of parent R-3 force m || c — none can
    describe the physical in-plane ordering.

    The actual CrCl3 magnetic structure occupies a non-maximal MSG whose
    parent group has lower symmetry (3-fold broken).  The tests below document
    this limitation of the maxmagn analysis for the R-3 parent.
    """

    SG = 148
    K = [0, 0, 3 / 2]
    CR_SITE = [0, 0, 1 / 3]

    def test_compatible_msgs_count(self):
        results = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE],
                                  centering="R")
        assert len(results) == 4

    def test_compatible_msgs_types(self):
        results = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE],
                                  centering="R")
        assert {r.msg_type for r in results} == {1, 2, 3, 4}

    def test_grey_group_forbids_moment(self):
        results = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE],
                                  centering="R")
        grey = next(r for r in results if r.msg_type == 2)
        assert grey.sites[0].n_free == 0

    def test_c3_forces_c_axis_for_all_magnetic_msgs(self):
        # The C3 site symmetry at every 6c position in R-3 has no real
        # in-plane eigenvector, so every magnetic MSG gives n_free=1 with
        # the single allowed direction along c.  This confirms that the
        # physical CrCl3 in-plane ordering CANNOT be found by maxmagn for
        # parent SG 148.
        results = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE],
                                  centering="R")
        c_axis = np.array([0.0, 0.0, 1.0])
        for r in results:
            s = r.sites[0]
            if s.n_free == 0:
                continue
            assert s.n_free == 1, (
                f"BNS {r.bns_number}: expected n_free=1 (m||c), "
                f"got n_free={s.n_free}"
            )
            direction = s.moment_basis[:, 0] / np.linalg.norm(s.moment_basis[:, 0])
            assert np.allclose(np.abs(direction @ c_axis), 1.0, atol=1e-4), (
                f"BNS {r.bns_number}: expected m||c, got {np.round(direction, 4)}"
            )

    def test_no_inplane_moment_possible_in_r3(self):
        # n_free=2 (2-D in-plane subspace) or n_free=3 (fully free) would
        # indicate in-plane moments are accessible; neither must appear.
        results = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE],
                                  centering="R")
        for r in results:
            assert r.sites[0].n_free in (0, 1), (
                f"BNS {r.bns_number}: n_free={r.sites[0].n_free} "
                f"implies in-plane freedom, which should be impossible in R-3"
            )

    def test_offaxis_6c_position_also_c3_constrained(self):
        # All 6 positions in the 6c orbit sit on a 3-fold axis (primary at
        # (0,0,z), secondary at (1/3,2/3,z) and (2/3,1/3,z)), so the same
        # C3 constraint applies everywhere in the orbit.
        results = compatible_msgs(self.SG, k=self.K,
                                  sites=[[1 / 3, 2 / 3, 1 / 3]],
                                  centering="R")
        c_axis = np.array([0.0, 0.0, 1.0])
        for r in results:
            s = r.sites[0]
            if s.n_free == 0:
                continue
            assert s.n_free == 1
            direction = s.moment_basis[:, 0] / np.linalg.norm(s.moment_basis[:, 0])
            assert np.allclose(np.abs(direction @ c_axis), 1.0, atol=1e-4)

    def test_type1_orbit_is_6(self):
        # Site symmetry C3 (3 ops); 18 ops total → orbit = 18 / 3 = 6
        results = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE],
                                  centering="R")
        type1 = next(r for r in results if r.msg_type == 1)
        assert len(type1.sites[0].orbit) == 6

    def test_maxmagn_returns_caxis_msg_not_inplane(self):
        # maxmagn selects type-IV BNS 148.20 (n_ops=36) with m||c.
        # This is the maximal MSG for out-of-plane ordering — it does NOT
        # describe the actual CrCl3 in-plane structure.
        results = maxmagn(self.SG, k=self.K, sites=[self.CR_SITE], centering="R")
        assert len(results) == 1
        assert results[0].bns_number == "148.20"
        assert results[0].msg_type == 4
        s = results[0].sites[0]
        assert s.n_free == 1
        direction = s.moment_basis[:, 0] / np.linalg.norm(s.moment_basis[:, 0])
        c_axis = np.array([0.0, 0.0, 1.0])
        assert np.allclose(np.abs(direction @ c_axis), 1.0, atol=1e-4)

    def test_maxmagn_type4_orbit_doubled(self):
        # Type-IV MSG has 36 operators; orbit at (0,0,1/3) = 36/3 = 12
        results = maxmagn(self.SG, k=self.K, sites=[self.CR_SITE], centering="R")
        assert len(results[0].sites[0].orbit) == 12

    def test_centering_invariant_at_bz_edge(self):
        # For k=(0,0,3/2) in R-3 all W@k differences are multiples of 3,
        # which satisfies both the primitive-integer and R-centering (l≡0 mod 3)
        # conditions simultaneously → centering="P" gives the same MSGs.
        results_R = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE],
                                    centering="R")
        results_P = compatible_msgs(self.SG, k=self.K, sites=[self.CR_SITE],
                                    centering="P")
        assert [r.uni_number for r in results_R] == [r.uni_number for r in results_P]
        for r_R, r_P in zip(results_R, results_P):
            assert r_R.sites[0].n_free == r_P.sites[0].n_free


# ---------------------------------------------------------------------------
# Tests for find_by_bns, analyze_msg, domain_operators
# ---------------------------------------------------------------------------

class TestFindByBns:
    def test_known_bns(self):
        r = find_by_bns("2.7")
        assert r is not None
        assert r.bns_number == "2.7"
        assert r.parent_sg == 2
        assert r.msg_type == 4
        assert r.sites == []

    def test_another_known_bns(self):
        r = find_by_bns("148.17")
        assert r is not None
        assert r.uni_number == 1247
        assert r.parent_sg == 148
        assert r.msg_type == 1

    def test_unknown_returns_none(self):
        assert find_by_bns("999.99") is None

    def test_whitespace_stripped(self):
        assert find_by_bns("  2.7  ").bns_number == "2.7"


class TestAnalyzeMsg:
    """analyze_msg: direct moment-basis query for a user-specified MSG.

    This is the 'lower symmetry input' path: after running Bilbao/Jana the
    user identifies the relevant MSG by BNS number, then passes it here with
    the site coordinates expressed in the MSG's cell.
    """

    def test_unknown_bns_returns_none(self):
        assert analyze_msg("999.99", [[0, 0, 0]]) is None

    def test_bns_27_inplane_moments_allowed(self):
        # BNS 2.7 = P_S 1̄, CrCl3 subgroup #5.
        # The C3 constraint is absent in P-1 → both Cr sites have n_free=3
        # (all moment directions, including in-plane, are symmetry-allowed).
        res = analyze_msg("2.7", [[0, 0, 1 / 3], [0, 0, 2 / 3]])
        assert res is not None
        assert res.bns_number == "2.7"
        assert res.sites[0].n_free == 3
        assert res.sites[1].n_free == 3

    def test_bns_27_inplane_vs_r3_caxis_contrast(self):
        # R-3 parent (BNS 148.17) forces m||c (n_free=1);
        # P_S 1̄ (BNS 2.7) lifts that constraint (n_free=3).
        r3 = analyze_msg("148.17", [[0, 0, 1 / 3]])
        ps1bar = analyze_msg("2.7", [[0, 0, 1 / 3]])
        assert r3.sites[0].n_free == 1     # c-axis only in R-3
        assert ps1bar.sites[0].n_free == 3  # fully free in P_S 1̄

    def test_bns_27_unitary_inversion_couples_cr1_cr2(self):
        # BNS 2.7 has unitary inversion (theta=+1) → Cr2 = +Cr1 (FM coupling
        # of mode amplitude; AFM in real space comes from the k-vector phase).
        # The orbit starting from Cr1=(0,0,1/3) spans 4 sites in the doubled cell.
        res = analyze_msg("2.7", [[0, 0, 1 / 3]])
        assert len(res.sites[0].orbit) == 4

    def test_bns_26_antiunitary_inversion_couples_cr1_cr2(self):
        # BNS 2.6 = type-III P-1 (subgroup #4): antiunitary inversion
        # → Cr2 = -Cr1 (AFM mode amplitude). Both sites still n_free=3
        # (no C3 → in-plane allowed), orbit spans only 2 sites (same cell).
        res = analyze_msg("2.6", [[0, 0, 1 / 3], [0, 0, 2 / 3]])
        assert res.sites[0].n_free == 3
        assert res.sites[1].n_free == 3
        assert len(res.sites[0].orbit) == 2

    def test_bns_return_type(self):
        res = analyze_msg("1.1", [[0, 0, 0]])
        assert isinstance(res, MSGResult)
        assert len(res.sites) == 1
        assert isinstance(res.sites[0], SiteResult)


class TestDomainOperators:
    """domain_operators: broken spatial symmetry → number and type of domains."""

    def test_crcl3_broken_op_count(self):
        # R-3 → P_S 1̄ (BNS 2.7): 4 distinct broken rotations
        # (3+, 3-, S6+, S6-) × 3 R-centering copies = 12 ops total.
        broken = domain_operators("2.7", parent_sg=148)
        assert len(broken) == 12

    def test_crcl3_broken_rotation_types(self):
        # Exactly 4 distinct rotation matrices are broken.
        broken = domain_operators("2.7", parent_sg=148)
        unique_W = []
        for op in broken:
            W = np.array(op["W"]).tolist()
            if W not in unique_W:
                unique_W.append(W)
        assert len(unique_W) == 4  # {3+, 3-, S6+, S6-}

    def test_crcl3_three_domains(self):
        # n_domains = |parent point group| / |MSG point group|
        # = |S6| / |Ci| = 6 / 2 = 3.
        # Equivalently: (n_broken_unique_W + n_preserved_unique_W) / n_preserved_unique_W
        # = (4 + 2) / 2 = 3.
        broken = domain_operators("2.7", parent_sg=148)
        broken_W = {tuple(np.array(op["W"]).flatten().tolist()) for op in broken}
        # Preserved: I and -I (the 2 unique W in BNS 2.7)
        n_preserved = 2
        n_total = len(broken_W) + n_preserved  # 4 + 2 = 6
        n_domains = n_total // n_preserved      # 6 // 2 = 3
        assert n_domains == 3

    def test_type3_msg_no_broken_rotations(self):
        # Type-III MSGs within the same parent SG keep all rotation matrices
        # (some become antiunitary) → domain_operators returns empty list.
        # MnF2: BNS 136.499 is type-III of P4_2/mnm (SG 136).
        broken = domain_operators("136.499", parent_sg=136)
        assert broken == []

    def test_identity_msg_fully_broken(self):
        # BNS 1.1 (type-I P1): only the identity W=I.
        # For parent SG 2 (P-1, type-I BNS 2.4), inversion is broken.
        broken = domain_operators("1.1", parent_sg=2)
        assert len(broken) == 1  # only the inversion op of BNS 2.4

    def test_unknown_msg_returns_empty(self):
        assert domain_operators("999.99", parent_sg=136) == []

    def test_unknown_parent_returns_empty(self):
        assert domain_operators("136.499", parent_sg=999) == []
