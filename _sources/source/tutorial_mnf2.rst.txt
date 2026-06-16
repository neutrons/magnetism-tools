MnF\ :sub:`2` — Maximal magnetic space group
============================================

MnF\ :sub:`2` (manganese fluoride) is a classic antiferromagnet with space
group :math:`P4_2/mnm` (No. 136), ordering wavevector **k** = (0, 0, 0), and
Mn\ :sup:`2+` at the Wyckoff 2a site (0, 0, 0).  This tutorial shows how to
use :func:`~msgjson.query.maximal_msgs` to identify the maximal MSGs
consistent with those symmetry constraints.

Setup
-----

.. code-block:: python

   import numpy as np
   from msgjson import maximal_msgs, compatible_msgs

   SG   = 136          # P4_2/mnm
   K    = [0, 0, 0]    # zone-centre ordering
   SITE = [0, 0, 0]    # Mn^2+ Wyckoff 2a

Finding all compatible MSGs
---------------------------

:func:`~msgjson.query.compatible_msgs` returns every MSG whose parent space
group is SG 136 and whose operators are compatible with **k** = 0.

.. code-block:: python

   results = compatible_msgs(SG, k=K, sites=[SITE])
   print(f"{len(results)} compatible MSGs for SG {SG}")

.. code-block:: text

   12 compatible MSGs for SG 136

Selecting maximal MSGs
----------------------

:func:`~msgjson.query.maximal_msgs` filters the list to those MSGs that
(i) allow a non-zero moment at the input site and (ii) have the highest
operator count — the maximal-symmetry magnetic orderings.

.. code-block:: python

   top = maximal_msgs(SG, k=K, sites=[SITE])

   for r in top:
       s = r.sites[0]
       v = s.moment_basis[:, 0]
       v /= np.linalg.norm(v)
       print(
           f"BNS {r.bns_number}  type={r.msg_type}"
           f"  n_ops={r.n_ops}  n_free={s.n_free}"
           f"  m || [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]"
       )

.. code-block:: text

   BNS 136.499  type=3  n_ops=16  n_free=1  m || [0.000, 0.000, 1.000]
   BNS 136.501  type=3  n_ops=16  n_free=1  m || [0.000, 0.000, 1.000]

Two type-III MSGs share the maximum operator count (16).  Both constrain
the Mn moment to lie along [0, 0, 1], consistent with the experimentally
observed easy-axis antiferromagnetism in MnF\ :sub:`2`.

Interpreting the results
------------------------

``n_free = 1``
    One free parameter — the magnitude of the moment along the allowed
    direction.  There are no in-plane components.

``type = 3``
    Type-III (black-white) MSGs: all rotation matrices of the parent SG are
    present, but some operations are antiunitary (paired with time reversal).
    This is the signature of a commensurate antiferromagnet with **k** = 0.

``orbit size = 2``
    The symmetry orbit of the 2a site contains 2 positions,
    (0, 0, 0) and (1/2, 1/2, 1/2), corresponding to the two Mn sublattices.

The two MSGs differ in which operations become antiunitary; both are valid
candidate descriptions and would be distinguished by the relative phase of
the moments on the two sublattice sites.
