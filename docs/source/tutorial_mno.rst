MnO — Type-II antiferromagnet on an FCC lattice
================================================

MnO (manganese oxide) is a textbook type-II antiferromagnet with the
rock-salt structure, space group :math:`Fm\bar{3}m` (No. 225).  The
Mn\ :sup:`2+` ions sit at the 4a Wyckoff site (0, 0, 0) and its
face-centred equivalents.  Below :math:`T_N \approx 118` K they order with
propagation vector **k** = (1/2, 1/2, 1/2), the L point of the FCC
Brillouin zone, stacking ferromagnetic (111) planes antiferromagnetically.

Setup
-----

.. code-block:: python

   import numpy as np
   from msgjson import maximal_msgs, subgroup_msgs

   SG      = 225              # Fm-3m
   K       = [0.5, 0.5, 0.5] # L point
   MN_SITE = [0, 0, 0]        # Mn^2+ at 4a

The sublattice structure
------------------------

The propagation vector **k** = (1/2, 1/2, 1/2) assigns a phase
:math:`e^{2\pi i\mathbf{k}\cdot\mathbf{r}}` to each site.  In the FCC
conventional cell the four Mn positions split 1:3 between the two
magnetic sublattices:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Site
     - **k** · **r**
     - Sublattice
   * - (0, 0, 0)
     - 0
     - \+
   * - (0, 1/2, 1/2)
     - 1/2
     - −
   * - (1/2, 0, 1/2)
     - 1/2
     - −
   * - (1/2, 1/2, 0)
     - 1/2
     - −

The (111) planes containing the "+" site alternate with those containing
three "−" sites, giving the characteristic layered AFM pattern.

Maximal MSG
-----------

.. code-block:: python

   top = maximal_msgs(SG, k=K, sites=[MN_SITE], centering="F")

   for r in top:
       s = r.sites[0]
       print(
           f"BNS {r.bns_number}  type={r.msg_type}  parent_sg={r.parent_sg}"
           f"  n_ops={r.n_ops}  n_free={s.n_free}"
           f"  orbit={len(s.orbit)}  origin={r.origin_shift}"
       )

.. code-block:: text

   BNS 2.7  type=4  parent_sg=2  n_ops=4  n_free=3  orbit=2  origin=(0.0, 0.0, 0.0)
   BNS 2.7  type=4  parent_sg=2  n_ops=4  n_free=3  orbit=2  origin=(0.0, 0.0, 0.5)

The result is BNS 2.7 (:math:`P_S\bar{1}`, parent SG 2), the same k-maximal
MSG returned for α-RuCl\ :sub:`3`.  This reflects a **basis-mismatch
limitation** of the subgroup-lattice search (identical to the one
documented in the subgroup section below): the true k-maximal MSGs for
Fm\ :math:`\bar{3}`\ m at **k** = (1/2, 1/2, 1/2) are the rhombohedral
R\ :sub:`I`\ -3m and R\ :sub:`I`\ -3c families [1]_, but expressing
their C\ :sub:`3` rotation in the *hexagonal* basis gives a different
W-matrix than C\ :sub:`3`\ [111] in the *cubic* basis.  The W-matrix
subset check cannot recognise them as equivalent, so only the
basis-independent P\ :math:`\bar{1}` / P1 families are found.

Bilbao's MAXMAGN [1]_ gives four k-maximal MSGs for this case
(two from the R\ :sub:`I`\ -3m branch and two from R\ :sub:`I`\ -3c,
one per conjugacy class), of which only one allows a nonzero spin on the
Mn atom at the origin.  Once the correct MSG is identified by Bilbao or
GSAS-II, :func:`~msgjson.query.analyze_msg` can compute the moment basis
for any candidate.

Subgroup lattice search
-----------------------

.. code-block:: python

   all_r = subgroup_msgs(SG, k=K, sites=[MN_SITE], centering="F")
   print(f"{len(all_r)} candidate MSGs")
   for r in all_r:
       s = r.sites[0]
       print(f"  BNS {r.bns_number:10s}  type={r.msg_type}  parent={r.parent_sg:3d}  n_free={s.n_free}")

.. code-block:: text

   7 candidate MSGs
     BNS 2.5        type=2  parent=  2  n_free=0
     BNS 2.7        type=4  parent=  2  n_free=3
     BNS 1.2        type=2  parent=  1  n_free=0
     BNS 1.3        type=4  parent=  1  n_free=3
     BNS 2.4        type=1  parent=  2  n_free=3
     BNS 2.6        type=3  parent=  2  n_free=0
     BNS 1.1        type=1  parent=  1  n_free=3

Only 7 candidates, all in the P\ :math:`\bar{1}` or P1 families.

The little co-group of **k** = (1/2, 1/2, 1/2) in :math:`Fm\bar{3}m` is
D\ :sub:`3d` — 12 unique rotation matrices including C\ :sub:`3`\ [111]
expressed in the *cubic* basis as

.. math::

   W_{C_3} = \begin{pmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}.

The physically relevant subgroups of :math:`Fm\bar{3}m` that allow
in-plane moments — such as the rhombohedral R\ :math:`\bar{3}` family
along [111] — also contain a three-fold axis, but express it in the
*hexagonal* basis:

.. math::

   W_{C_3}^{\text{hex}} = \begin{pmatrix} -1 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}.

These two matrices represent the same geometric rotation in different
coordinate systems; the current W-matrix comparison in
:func:`~msgjson.query.subgroup_msgs` does not perform the basis
transformation needed to recognise them as equivalent.  As a result, the
trigonal/rhombohedral subgroups are absent from the search results, and
only the trivially-basis-independent P\ :math:`\bar{1}` and P1 families
are recovered.

For MnO, the full subgroup analysis including the proper rhombohedral
treatment is available through the Bilbao k-SUBGROUPSMAG tool [1]_ or
directly in GSAS-II [2]_ and Jana2020 [4]_.  Once the MSG is identified by
those tools, :func:`~msgjson.query.analyze_msg` can be used to compute the
moment basis for any specific candidate.

Comparison of the four examples
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * -
     - MnF\ :sub:`2`
     - CrCl\ :sub:`3`
     - α-RuCl\ :sub:`3`
     - MnO
   * - Parent SG
     - 136 (P4\ :sub:`2`/mnm)
     - 148 (R\ :math:`\bar{3}`)
     - 148 (R\ :math:`\bar{3}`)
     - 225 (Fm\ :math:`\bar{3}`\ m)
   * - **k**
     - (0, 0, 0)
     - (0, 0, 3/2)
     - (0, 1/2, 1)
     - (1/2, 1/2, 1/2)
   * - k-maximal MSG (BNS)
     - 136.499 / 136.501
     - 148.20
     - **2.7** (parent 2)
     - **2.7** (basis mismatch)
   * - Maximal MSG n_free
     - 1 (m ‖ c)
     - 1 (m ‖ c)
     - 3 (free)
     - 3 (free, basis mismatch)
   * - Maximal origin variants
     - 1
     - 2
     - 2
     - 2
   * - Subgroup candidates
     - 12 (parent only)
     - 27 (crosses SGs)
     - 7 (P-1/P1 only)
     - 7 (P-1/P1 only)
   * - Physical m
     - ‖ c ✓ maximal
     - ⊥ c — non-maximal
     - in-plane — already free
     - ⊥ [111] — non-maximal
   * - Subgroup limitation
     - None
     - Trigonal → triclinic ✓
     - Trigonal → triclinic ✓
     - Cubic → trigonal ✗

.. [1] J. M. Perez-Mato et al., *Annu. Rev. Mater. Res.* **45**, 217 (2015).
.. [2] R. B. Von Dreele and L. Elcoro, *Acta Cryst. B* **80** (2024).
.. [4] M. S. Henriques et al., *Acta Cryst. B* **80** (2024).
