CrCl\ :sub:`3` — Zigzag antiferromagnet
========================================

CrCl\ :sub:`3` orders below :math:`T_N \approx 14` K in a zigzag
antiferromagnetic pattern with Cr\ :sup:`3+` moments in the ab-plane [1]_.
Space group :math:`R\bar{3}` (No. 148), propagation vector
**k** = (0, 0, 3/2) in the hexagonal setting.  The MSG from diffraction
refinement is BNS 2.7 (:math:`P_S\bar{1}`, type IV).  The k-maximal MSG of
R\ :math:`\bar{3}` (BNS 148.20) forces **m** ‖ **c** due to the C\ :sub:`3`
site symmetry, so the in-plane ordering is non-maximal and is only recovered
by a subgroup search.

Setup
-----

.. code-block:: python

   import numpy as np
   from msgjson import analyze_msg, orbit_moments, magnetic_structure_factors

   K        = [0, 0, 3/2]
   CR_SITES = [[0, 0, 1/3], [0, 0, 2/3]]   # Cr^3+ Wyckoff 6c

Magnetic space group
--------------------

BNS 2.7 is identified by refinement.  Two domain states exist: the
anti-translation τ = (0, 0, ½) can be placed at origin_shift = 0 or 0.5;
both give the same :math:`|F_M|^2`.

.. code-block:: python

   res = analyze_msg("2.7", CR_SITES)

   for i, s in enumerate(res.sites):
       print(f"  Cr{i+1} at {s.site}:  n_free = {s.n_free}")

.. code-block:: text

     Cr1 at [0.  0.  0.333]:  n_free = 3
     Cr2 at [0.  0.  0.667]:  n_free = 3

``n_free = 3`` — any moment direction is symmetry-allowed; the in-plane
orientation is selected by exchange anisotropy.

Moment orbit
------------

.. code-block:: python

   m_amp = 3.0                          # μ_B, Cr³⁺ ordered moment
   m_ref = np.array([m_amp, 0.0, 0.0])  # along hexagonal a*

   site_moms = orbit_moments(res, site_idx=0, m_ref=m_ref, k=K, centering="R")

   for pos, mom in site_moms:
       print(f"  {np.round(pos, 4)}:  m = {np.round(mom, 2)} μ_B")

.. code-block:: text

     [0.     0.     0.3333]:  m = [3. 0. 0.] μ_B
     [0.     0.     0.6667]:  m = [3. 0. 0.] μ_B
     [0.     0.     0.8333]:  m = [-3.  0.  0.] μ_B
     [0.     0.     0.1667]:  m = [-3.  0.  0.] μ_B

Four-site zigzag +,+,−,− along *c*.  Net moment zero (antiferromagnet).

Structure factors
-----------------

Satellites appear at **Q** = **τ** + **k** where **τ** satisfies the
R-centering condition −h + k + l ≡ 0 (mod 3):

.. code-block:: python

   a, c_lat = 5.963, 17.28
   lattice = np.array([[a,    0,              0    ],
                       [-a/2, a*np.sqrt(3)/2, 0    ],
                       [0,    0,              c_lat]])
   B = 2 * np.pi * np.linalg.inv(lattice).T

   hkl_list = [(0,0,0), (0,0,3), (2,-1,0), (-1,2,0)]
   F2 = magnetic_structure_factors(
       site_moms, hkl_list, K, lattice=lattice, ion="Cr3+"
   )

   print(f"  {'τ':12s}  {'Q = τ+k':12s}  {'|F_M|²':>8s}  {'sin²α':>6s}  {'|F_M⊥|²':>9s}")
   for hkl_i, f2 in zip(hkl_list, F2):
       Q_frac = tuple(h + ki for h, ki in zip(hkl_i, K))
       Q_cart = B @ np.array(Q_frac)
       Q_mag  = np.linalg.norm(Q_cart)
       s2     = 1 - np.dot(Q_cart / Q_mag, m_ref / np.linalg.norm(m_ref))**2
       qstr   = "({:.0f},{:.0f},{:.1f})".format(*Q_frac)
       print(f"  {str(hkl_i):12s}  {qstr:12s}  {f2:8.2f}  {s2:6.3f}  {f2*s2:9.2f}")

.. code-block:: text

     τ             Q = τ+k        |F_M|²  sin²α    |F_M⊥|²
     (0, 0, 0)     (0,0,1.5)       34.61   1.000      34.61
     (0, 0, 3)     (0,0,4.5)       25.36   1.000      25.36
     (2, -1, 0)    (2,-1,1.5)      21.36   0.442       9.43
     (-1, 2, 0)    (-1,2,1.5)      16.20   0.996      16.13

The (0, 0, 3/2) and (0, 0, 9/2) satellites lie along **c** so
sin²α = 1 for in-plane moments — these are the most sensitive probes
of the ordering.

.. [1] M. A. McGuire et al.,
   *Magnetic behavior and spin-lattice coupling in cleavable van der Waals
   layered CrCl*\ :sub:`3` *crystals*,
   Phys. Rev. Materials **1**, 014001 (2017).
