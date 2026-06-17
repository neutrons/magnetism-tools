MnF\ :sub:`2` — Collinear antiferromagnet
=========================================

MnF\ :sub:`2` orders below :math:`T_N = 67` K as a two-sublattice
antiferromagnet with Mn\ :sup:`2+` moments along the tetragonal *c*-axis [1]_.
Space group :math:`P4_2/mnm` (No. 136), propagation vector **k** = (0, 0, 0):
magnetic satellite peaks coincide with nuclear Bragg positions.

Setup
-----

.. code-block:: python

   import numpy as np
   from msgjson import maximal_msgs, orbit_moments, magnetic_structure_factors

   SG   = 136
   K    = [0, 0, 0]
   SITE = [0, 0, 0]   # Mn^2+ Wyckoff 2a

Magnetic space group
--------------------

.. code-block:: python

   top = maximal_msgs(SG, k=K, sites=[SITE])

   for r in top:
       s = r.sites[0]
       v = s.moment_basis[:, 0] / np.linalg.norm(s.moment_basis[:, 0])
       print(f"BNS {r.bns_number}  type={r.msg_type}  n_ops={r.n_ops}"
             f"  n_free={s.n_free}  m || [{v[0]:.0f},{v[1]:.0f},{v[2]:.0f}]")

.. code-block:: text

   BNS 136.499  type=3  n_ops=16  n_free=1  m || [0,0,1]
   BNS 136.501  type=3  n_ops=16  n_free=1  m || [0,0,1]

Both type-III MSGs force **m** ‖ **c** and represent the same observable
structure at **k** = 0.  No structural domains arise from this ordering.

Moment orbit
------------

.. code-block:: python

   r = top[0]   # BNS 136.499
   s = r.sites[0]

   m_amp = 4.9                           # μ_B, measured Mn²⁺ ordered moment
   m_ref = m_amp * s.moment_basis[:, 0]  # [0., 0., 4.9]

   site_moms = orbit_moments(r, site_idx=0, m_ref=m_ref, k=K)

   for pos, mom in site_moms:
       print(f"  {pos}:  m = {np.round(mom, 2)} μ_B")

.. code-block:: text

     [0.  0.  0.]:  m = [0.  0.  4.9] μ_B
     [0.5 0.5 0.5]:  m = [0.  0. -4.9] μ_B

Two-sublattice antiferromagnet; net moment zero.

Structure factors
-----------------

For **k** = 0 the satellite positions coincide with integer *hkl*.
Passing the tetragonal lattice parameters activates the Mn\ :sup:`2+`
form factor:

.. code-block:: python

   lattice = np.diag([4.873, 4.873, 3.306])   # Å, tetragonal

   hkl = [(1,0,0), (0,0,1), (1,0,1), (1,1,0), (1,1,1), (2,0,1), (2,1,0)]
   F2 = magnetic_structure_factors(
       site_moms, hkl, K, lattice=lattice, ion="Mn2+"
   )

   print(f"  {'hkl':12s}  |F_M|² (μ_B²)")
   for idx, f2 in zip(hkl, F2):
       print(f"  {str(idx):12s}  {f2:10.2f}")

.. code-block:: text

     hkl           |F_M|² (μ_B²)
     (1, 0, 0)          76.26
     (0, 0, 1)          58.81
     (1, 0, 1)           0.00
     (1, 1, 0)           0.00
     (1, 1, 1)          38.54
     (2, 0, 1)          25.86
     (2, 1, 0)          32.59

Selection rule: :math:`F_M \propto 1 - e^{i\pi(h+k+l)}` — zero when
*h* + *k* + *l* is even, active when odd.  Although (0, 0, 1) has a
nonzero :math:`|F_M|^2`, the moment is parallel to **Q** (sin²α = 0)
so the peak carries no neutron cross-section.

.. [1] Z. Yamani, Z. Tun, and D. H. Ryan,
   *Neutron scattering study of the classical antiferromagnet MnF*\ :sub:`2`\ *:
   a perfect hands-on neutron scattering teaching course*,
   Can. J. Phys. **88**, 771 (2010).
