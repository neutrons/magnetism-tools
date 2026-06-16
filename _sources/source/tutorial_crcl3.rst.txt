CrCl\ :sub:`3` — Subgroup lattice search for non-maximal ordering
==================================================================

CrCl\ :sub:`3` (chromium trichloride) has space group :math:`R\bar{3}`
(No. 148) and orders antiferromagnetically along **c** with in-plane moments
at a propagation vector **k** = (0, 0, 3/2) in the hexagonal setting.  The
Cr\ :sup:`3+` ions occupy the 6c Wyckoff site.

This tutorial shows that the physical in-plane ordering is *not* accessible
from the maximal MSGs of R-3 alone, and how
:func:`~msgjson.query.subgroup_msgs` recovers it by searching the full
subgroup lattice.

Setup
-----

.. code-block:: python

   import numpy as np
   from msgjson import maximal_msgs, subgroup_msgs, analyze_msg, domain_operators

   SG        = 148              # R-3
   K         = [0, 0, 3/2]     # BZ edge along c* (R centering: l = 3/2 allowed)
   CR_SITES  = [[0, 0, 1/3],   # Cr^3+ at 6c (z ≈ 1/3)
                [0, 0, 2/3]]

Step 1 — Maximal MSGs of the parent space group
------------------------------------------------

Starting from the parent SG 148 and R centering:

.. code-block:: python

   top = maximal_msgs(SG, k=K, sites=CR_SITES, centering="R")

   for r in top:
       s = r.sites[0]
       v = s.moment_basis[:, 0] / np.linalg.norm(s.moment_basis[:, 0])
       print(
           f"BNS {r.bns_number}  type={r.msg_type}"
           f"  n_ops={r.n_ops}  n_free={s.n_free}"
           f"  origin={r.origin_shift}"
           f"  m || [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]"
       )

.. code-block:: text

   BNS 148.20  type=4  n_ops=36  n_free=1  origin=(0.0, 0.0, 0.0)  m || [0.000, 0.000, 1.000]
   BNS 148.20  type=4  n_ops=36  n_free=1  origin=(0.0, 0.0, 0.5)  m || [0.000, 0.000, 1.000]

Two results share the same BNS number 148.20 but differ in their
``origin_shift`` — the position of the MSG origin in the parent cell.
They represent the two inequivalent **domain states** arising from the
type-IV anti-translation τ = (0, 0, 1/2): placing the origin at **p** = 0
or at **p** = τ produces physically distinct magnetic structures related by
a time-reversal operation.  Both force **m** ‖ **c**, a direct consequence
of the C\ :sub:`3` site symmetry: every position in the 6c orbit lies on a
three-fold axis, which has no real in-plane eigenvector.

The physical CrCl\ :sub:`3` structure has in-plane moments — it cannot be
described by any maximal MSG of the R-3 parent.

Step 2 — Full subgroup lattice
------------------------------

:func:`~msgjson.query.subgroup_msgs` searches all 1651 MSGs for those whose
rotation matrices lie within the little co-group of **k** in SG 148.  This
crosses parent-SG boundaries and descends all the way to MSG 1.1.

.. code-block:: python

   all_r = subgroup_msgs(SG, k=K, sites=CR_SITES, centering="R")
   print(f"{len(all_r)} candidate MSGs\n")

   print(f"{'BNS':10s}  type  parent  n_ops  n_free(Cr1)  n_free(Cr2)")
   for r in all_r:
       nf1, nf2 = r.sites[0].n_free, r.sites[1].n_free
       print(
           f"  {r.bns_number:8s}    {r.msg_type}     {r.parent_sg:3d}"
           f"     {r.n_ops:2d}       {nf1}          {nf2}"
       )

.. code-block:: text

   27 candidate MSGs

   BNS         type  parent  n_ops  n_free(Cr1)  n_free(Cr2)
     148.18      2     148     36       0          0
     148.20      4     148     36       1          1
     146.11      2     146     18       0          0
     146.12      4     146     18       1          1
     148.17      1     148     18       1          1
     148.19      3     148     18       1          1
     147.14      2     147     12       0          0
     147.16      4     147     12       1          1
     146.10      1     146      9       1          1
     143.2       2     143      6       0          0
     143.3       4     143      6       1          1
     144.5       2     144      6       0          0
     144.6       4     144      6       3          3
     145.8       2     145      6       0          0
     145.9       4     145      6       3          3
     147.13      1     147      6       1          1
     147.15      3     147      6       1          1
     2.5         2       2      4       0          0
     2.7         4       2      4       3          3
     143.1       1     143      3       1          1
     144.4       1     144      3       3          3
     145.7       1     145      3       3          3
     1.2         2       1      2       0          0
     1.3         4       1      2       3          3
     2.4         1       2      2       3          3
     2.6         3       2      2       3          3
     1.1         1       1      1       3          3

MSGs with ``n_free = 3`` (both sites) allow fully general moments, including
in-plane directions.  They appear once the three-fold symmetry is broken,
i.e. for parent SGs that lack a C\ :sub:`3` axis: P-1 (parent 2) and P1
(parent 1).

Step 3 — Analysing the physical structure (BNS 2.7)
----------------------------------------------------

From diffraction refinement (e.g. in Jana2020 or GSAS-II) the
in-plane AFM stacking is described by BNS 2.7 (:math:`P_S\bar{1}`, type IV).
Use :func:`~msgjson.query.analyze_msg` to confirm the moment freedom without
re-running the k-filter:

.. code-block:: python

   res = analyze_msg("2.7", CR_SITES)

   for i, s in enumerate(res.sites):
       print(f"Cr{i+1} at {s.site}:  n_free = {s.n_free}")

.. code-block:: text

   Cr1 at [0.  0.  0.333]:  n_free = 3
   Cr2 at [0.  0.  0.667]:  n_free = 3

Both sites are unconstrained — any moment direction is symmetry-allowed,
consistent with the observed in-plane orientation.

Step 4 — Magnetic domains
-------------------------

The descent from R-3 (point group S\ :sub:`6`, order 6) to P\ :sub:`S`\
:math:`\bar{1}` (point group C\ :sub:`i`, order 2) breaks the three-fold
rotation, generating three equivalent domains:

.. code-block:: python

   broken = domain_operators("2.7", parent_sg=SG)

   unique_W = []
   for op in broken:
       W = np.array(op["W"]).tolist()
       if W not in unique_W:
           unique_W.append(W)

   n_domains = (len(unique_W) + 2) // 2   # (4 broken + 2 preserved) / 2
   print(f"{len(broken)} broken operators  ({len(unique_W)} unique rotations)")
   print(f"n_domains = |S6| / |Ci| = 6 / 2 = {n_domains}")

.. code-block:: text

   12 broken operators  (4 unique rotations)
   n_domains = |S6| / |Ci| = 6 / 2 = 3

The four broken rotation matrices are the C\ :sub:`3`\ :sup:`+`,
C\ :sub:`3`\ :sup:`-`, S\ :sub:`6`\ :sup:`+`, and S\ :sub:`6`\ :sup:`-`
operations, each appearing three times due to R centering.
