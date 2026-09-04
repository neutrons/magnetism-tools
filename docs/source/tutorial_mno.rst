MnO — Type-II antiferromagnet
==============================

MnO orders below :math:`T_N = 118` K with Mn\ :sup:`2+` moments in {111}
planes coupled antiferromagnetically along [111] [3]_.  Space group
:math:`Fm\bar{3}m` (No. 225), propagation vector **k** = (½, ½, ½).  The
correct k-maximal MSG belongs to the R\ :sub:`I`-3m family (Bilbao notation);
the present library returns only BNS 2.7 (P\ :math:`\bar{1}`) for this case
because the C\ :sub:`3`\ [111] axis in the cubic basis is not recognised as
equivalent to the hexagonal C\ :sub:`3` in the library's W-matrix comparison.
Use :func:`~msgjson.query.analyze_msg` with the BNS number from MAXMAGN or
your refinement software once the MSG is known.

Setup
-----

.. code-block:: python

   import numpy as np
   from msgjson import magnetic_structure_factors, b_matrix

   K = [0.5, 0.5, 0.5]   # L point, FCC Brillouin zone

FCC conventional cell — four Mn sites
--------------------------------------

With **k** = (½, ½, ½) the phase :math:`e^{2\pi i\mathbf{k}\cdot\mathbf{r}}`
is +1 at (0, 0, 0) and −1 at the three face-centred positions.  Total neutron
scattering refinement [3]_ establishes moments along ⟨11̄2̄⟩; here we take
**m** ‖ [1, 1, −2]:

.. code-block:: python

   m_amp = 4.58                                    # μ_B, measured Mn²⁺ moment
   m_dir = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
   m = m_amp * m_dir

   site_moms = [
       (np.array([0.0,  0.0,  0.0 ]),  +m),   # phase +1
       (np.array([0.0,  0.5,  0.5 ]), -m),    # phase −1
       (np.array([0.5,  0.0,  0.5 ]), -m),    # phase −1
       (np.array([0.5,  0.5,  0.0 ]), -m),    # phase −1
   ]

Three-to-one sublattice split (type-II AFM); net moment zero.

Structure factors
-----------------

Satellites at **Q** = **τ** + **k** for FCC-allowed **τ** (h, k, l all
even or all odd):

.. code-block:: python

   a = 4.445
   lattice = a * np.eye(3)
   B = b_matrix(lattice)

   hkl_list = [(0,0,0), (1,1,1), (1,1,-1), (-1,1,1), (2,0,0), (0,0,2)]
   F2 = magnetic_structure_factors(
       site_moms, hkl_list, K, lattice=lattice, ion="Mn2+"
   )

   print(f"  {'τ':12s}  {'Q = τ+k':20s}  {'|F_M|²':>8s}  {'sin²α':>6s}  {'|F_M⊥|²':>9s}")
   for hkl_i, f2 in zip(hkl_list, F2):
       Q_frac = tuple(h + ki for h, ki in zip(hkl_i, K))
       Q_cart = B @ np.array(Q_frac)
       Q_mag  = np.linalg.norm(Q_cart)
       s2     = 1 - np.dot(Q_cart / Q_mag, m_dir)**2
       qstr   = "({:.1f},{:.1f},{:.1f})".format(*Q_frac)
       print(f"  {str(hkl_i):12s}  {qstr:20s}  {f2:8.3f}  {s2:6.3f}  {f2*s2:9.3f}")

.. code-block:: text

     τ             Q = τ+k                 |F_M|²   sin²α    |F_M⊥|²
     (0, 0, 0)     (0.5,0.5,0.5)          272.508   1.000    272.508
     (1, 1, 1)     (1.5,1.5,1.5)           62.626   1.000     62.626
     (1, 1, -1)    (1.5,1.5,-0.5)          98.939   0.439     43.394
     (-1, 1, 1)    (-0.5,1.5,1.5)          98.939   0.860     85.053
     (2, 0, 0)     (2.5,0.5,0.5)           62.626   0.901     56.441
     (0, 0, 2)     (0.5,0.5,2.5)           62.626   0.605     37.885

All FCC-allowed **τ** give constructive interference (phases 1, −1, −1, −1
sum to 4 for the 1:3 sublattice split), so the only intensity variation
comes from the Mn\ :sup:`2+` form factor and sin²α.  The strong variation in
sin²α across symmetry-equivalent :math:`|\mathbf{Q}|` peaks
(e.g. (1, 1, −1) vs (−1, 1, 1): 0.439 vs 0.860) directly encodes the moment
orientation within the {111} plane and is what the Goodwin total-scattering
refinement used to establish the ⟨11̄2̄⟩ direction.

.. [3] A. L. Goodwin et al.,
   *Magnetic structure of MnO at 10 K from total neutron scattering data*,
   Phys. Rev. Lett. **96**, 047209 (2006).
