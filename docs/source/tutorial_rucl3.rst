α-RuCl\ :sub:`3` — Zigzag antiferromagnet, k = (0, ½, 1)
==========================================================

α-RuCl\ :sub:`3` orders below :math:`T_N \approx 7` K in a zigzag
antiferromagnetic pattern with Ru\ :sup:`3+` moments in the ab-plane [2]_.
Space group :math:`R\bar{3}` (No. 148), propagation vector
**k** = (0, ½, 1) (hexagonal setting).  The k-maximal MSG is already
BNS 2.7 (:math:`P_S\bar{1}`, type IV) — no symmetry breaking beyond the
maximal is required to allow in-plane moments.

Setup
-----

.. code-block:: python

   import numpy as np
   from msgjson import maximal_msgs, orbit_moments, magnetic_structure_factors

   SG      = 148
   K       = [0, 0.5, 1.0]
   RU_SITE = [0, 0, 1/3]   # Ru^3+ Wyckoff 6c

Magnetic space group
--------------------

.. code-block:: python

   top = maximal_msgs(SG, k=K, sites=[RU_SITE], centering="R")

   for r in top:
       s = r.sites[0]
       print(f"BNS {r.bns_number}  type={r.msg_type}  n_ops={r.n_ops}"
             f"  n_free={s.n_free}  origin={r.origin_shift}")

.. code-block:: text

   BNS 2.7  type=4  n_ops=4  n_free=3  origin=(0.0, 0.0, 0.0)
   BNS 2.7  type=4  n_ops=4  n_free=3  origin=(0.0, 0.0, 0.5)

Two domain states arising from the two inequivalent positions of the
anti-translation τ = (0, 0, ½).  Both give the same :math:`|F_M|^2`.
``n_free = 3`` — any moment direction is symmetry-allowed.

Moment orbit
------------

.. code-block:: python

   r = top[0]
   m_amp = 0.5                          # μ_B, Ru³⁺ ordered moment
   m_ref = m_amp * np.array([1.0, 0.0, 0.0])   # in-plane, along a*

   site_moms = orbit_moments(r, site_idx=0, m_ref=m_ref, k=K, centering="R")

   for pos, mom in site_moms:
       print(f"  {np.round(pos, 4)}:  m = {np.round(mom, 3)} μ_B")

.. code-block:: text

     [0.     0.     0.3333]:  m = [0.5 0.  0. ] μ_B
     [0.     0.     0.6667]:  m = [0.5 0.  0. ] μ_B
     [0.     0.     0.8333]:  m = [-0.5  0.   0. ] μ_B
     [0.     0.     0.1667]:  m = [-0.5  0.   0. ] μ_B

Four-site zigzag +,+,−,− along *c*.  Net moment zero (antiferromagnet).

Structure factors
-----------------

Satellites appear at **Q** = **τ** + **k** where **τ** satisfies the
R-centering condition −h + k + l ≡ 0 (mod 3):

.. code-block:: python

   a, c_lat = 5.98, 17.0
   lattice = np.array([[a,    0,              0    ],
                       [-a/2, a*np.sqrt(3)/2, 0    ],
                       [0,    0,              c_lat]])
   B = 2 * np.pi * np.linalg.inv(lattice).T

   hkl_list = [(0,0,0), (0,1,-1), (0,0,3), (0,0,6)]
   F2 = magnetic_structure_factors(
       site_moms, hkl_list, K, lattice=lattice, ion="Ru1+"
   )

   print(f"  {'τ':12s}  {'Q = τ+k':18s}  {'|F_M|²':>8s}  {'sin²α':>6s}  {'|F_M⊥|²':>9s}")
   for hkl_i, f2 in zip(hkl_list, F2):
       Q_frac  = tuple(h + ki for h, ki in zip(hkl_i, K))
       Q_cart  = B @ np.array(Q_frac)
       Q_mag   = np.linalg.norm(Q_cart)
       s2      = 1 - np.dot(Q_cart / Q_mag, m_ref / np.linalg.norm(m_ref))**2
       qstr    = "({:.0f},{:.1f},{:.0f})".format(*Q_frac)
       print(f"  {str(hkl_i):12s}  {qstr:18s}  {f2:8.3f}  {s2:6.3f}  {f2*s2:9.3f}")

.. code-block:: text

     τ             Q = τ+k              |F_M|²   sin²α    |F_M⊥|²
     (0, 0, 0)     (0,0.5,1)             0.844   0.846      0.714
     (0, 1, -1)    (0,1.5,0)             0.000   0.800      0.000
     (0, 0, 3)     (0,0.5,4)             0.000   0.965      0.000
     (0, 0, 6)     (0,0.5,7)             0.147   0.987      0.145

Two independent extinctions reduce the observable peak count: (0, 1.5, 0)
cancels because Q\ :sub:`z` = 0 makes all four orbit phases equal; (0, 0.5, 4)
cancels from the +,+,−,− phase combination at Q\ :sub:`z` = 4.  The
satellite at (0, ½, 1) from τ = (0, 0, 0) is the primary experimental
target.

.. [2] S.-Y. Park et al.,
   *Emergence of the isotropic Kitaev honeycomb lattice α-RuCl*\ :sub:`3`
   *and its magnetic properties*,
   J. Phys.: Condens. Matter **36**, 215803 (2024).
