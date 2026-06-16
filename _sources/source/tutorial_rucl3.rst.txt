α-RuCl\ :sub:`3` — Zigzag order and the role of the k-vector
=============================================================

α-RuCl\ :sub:`3` adopts the R\ :math:`\bar{3}` structure (No. 148) at low
temperature and orders in a zigzag pattern with propagation vector
**k** = (0, 1/2, 1) in the hexagonal setting.  The Ru\ :sup:`3+` ions
occupy the 6c Wyckoff site.

This tutorial shows how the choice of **k** affects the symmetry analysis,
producing a qualitatively different result from CrCl\ :sub:`3` despite both
materials sharing the same parent space group.

Setup
-----

.. code-block:: python

   import numpy as np
   from msgjson import maximal_msgs, subgroup_msgs, analyze_msg

   SG      = 148              # R-3
   K       = [0, 0.5, 1.0]   # zigzag propagation vector (hexagonal setting)
   RU_SITE = [0, 0, 1/3]     # Ru^3+ at 6c

The little co-group of **k**
-----------------------------

The first step is to identify which rotation matrices map **k** to ±**k**
modulo the R-centered reciprocal lattice.  For CrCl\ :sub:`3` with
**k** = (0, 0, 3/2) the entire S\ :sub:`6` point group (all six rotations
of R\ :math:`\bar{3}`) was k-compatible.  For α-RuCl\ :sub:`3` the
situation is different:

- The C\ :sub:`3` rotation maps (0, 1/2, 1) → (1/2, 0, 1), which
  is **not** equivalent to ±**k** modulo the R lattice.
- Only the identity **E** and inversion **i** satisfy the k-compatibility
  condition.

The little co-group is therefore C\ :sub:`i` = {**E**, **i**}, a strict
subgroup of S\ :sub:`6`.  This has direct consequences for moment symmetry.

Maximal MSGs
------------

.. code-block:: python

   top = maximal_msgs(SG, k=K, sites=[RU_SITE], centering="R")

   for r in top:
       s = r.sites[0]
       print(
           f"BNS {r.bns_number}  type={r.msg_type}  parent_sg={r.parent_sg}"
           f"  n_ops={r.n_ops}  n_free={s.n_free}"
           f"  orbit={len(s.orbit)}  origin={r.origin_shift}"
       )

.. code-block:: text

   BNS 2.7  type=4  parent_sg=2  n_ops=4  n_free=3  orbit=4  origin=(0.0, 0.0, 0.0)
   BNS 2.7  type=4  parent_sg=2  n_ops=4  n_free=3  orbit=4  origin=(0.0, 0.0, 0.5)

The k-maximal MSG is BNS 2.7 (:math:`P_S\bar{1}`, type IV), whose parent
space group is SG 2 (P\ :math:`\bar{1}`) — **not** the input parent SG 148
(R\ :math:`\bar{3}`).  This is the defining feature of the Bilbao MAXMAGN
definition [1]_: a MSG is k-maximal if it has *no k-compatible supergroup*
anywhere in the subgroup lattice of G1'.  Because the little co-group of
**k** = (0, 1/2, 1) contains only W ∈ {**E**, **i**}, the entire R-3
operator set lies outside the little co-group.  No MSG belonging to
SG 148 can be k-compatible — the k-maximal MSG therefore crosses the
parent-SG boundary to the P\ :math:`\bar{1}` family.

The two results again represent inequivalent origin placements (domain
states) of the type-IV anti-translation τ = (0, 0, 1/2).

n_free = 3 for both: with no C\ :sub:`3` constraint anywhere in the
P\ :math:`\bar{1}` symmetry, all moment directions — including in-plane
— are already allowed at the maximal level.  This contrasts with
CrCl\ :sub:`3`, where the k-maximal MSG (also BNS 148.20) imposed m ‖ c.

Subgroup lattice
----------------

.. code-block:: python

   all_r = subgroup_msgs(SG, k=K, sites=[RU_SITE], centering="R")
   print(f"{len(all_r)} candidate MSGs\n")

   print(f"{'BNS':10s}  type  parent  n_ops  n_free")
   for r in all_r:
       s = r.sites[0]
       print(
           f"  {r.bns_number:8s}    {r.msg_type}     {r.parent_sg:3d}"
           f"     {r.n_ops:2d}       {s.n_free}"
       )

.. code-block:: text

   7 candidate MSGs

   BNS         type  parent  n_ops  n_free
     2.5         2       2      4       0
     2.7         4       2      4       3
     1.2         2       1      2       0
     1.3         4       1      2       3
     2.4         1       2      2       3
     2.6         3       2      2       3
     1.1         1       1      1       3

Only 7 candidates compared to 27 for CrCl\ :sub:`3`.  Because the little
co-group contains only W ∈ {**E**, **i**}, the subgroup lattice is
exhausted by the P\ :math:`\bar{1}` (parent 2) and P1 (parent 1) families.
All MSGs from SG 148 itself are absent — the R-3 rotations (C\ :sub:`3`,
S\ :sub:`6`) are not k-compatible with **k** = (0, 1/2, 1) and so
cannot appear as operators in any candidate MSG.

Contrast with CrCl\ :sub:`3`
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - CrCl\ :sub:`3`
     - α-RuCl\ :sub:`3`
   * - **k**
     - (0, 0, 3/2)
     - (0, 1/2, 1)
   * - Little co-group of **k**
     - S\ :sub:`6` (full R-3 point group)
     - C\ :sub:`i` (inversion only)
   * - k-maximal MSG (BNS)
     - 148.20 (parent 148, R-3)
     - **2.7** (parent 2, P\ :math:`\bar{1}`)
   * - Maximal MSG n_free
     - 1 (m ‖ c only)
     - 3 (fully free)
   * - Maximal origin variants
     - 2 (type-IV τ)
     - 2 (type-IV τ)
   * - Subgroup candidates
     - 27
     - 7
   * - In-plane accessible from maximal?
     - No — requires symmetry lowering
     - Yes — already at maximal level

The key difference is not the parent space group but the propagation vector:
**k** = (0, 0, 3/2) preserves the three-fold axis in the little co-group
so the k-maximal MSG stays within the R-3 family; **k** = (0, 1/2, 1)
expels C\ :sub:`3` from the little co-group entirely, driving the k-maximal
MSG to the P\ :math:`\bar{1}` family and lifting the c-axis constraint from
the outset.

.. [1] J. M. Perez-Mato et al., *Annu. Rev. Mater. Res.* **45**, 217 (2015).
