msgjson documentation
=====================

*msgjson* provides magnetic space group (MSG) operator tables for all 1651
commensurate Shubnikov groups and implements symmetry-analysis workflows for
magnetic structures.  The library reproduces the k-SUBGROUPSMAG
[PerezMato2015]_ and MAXMAGN workflows from the Bilbao Crystallographic
Server [VonDreele2024]_, and is designed to complement magnetic structure
refinement in FullProf [RodriguezCarvajal2021]_ and Jana2020 [Henriques2024]_.

Features
--------

- Operator tables for all 1651 commensurate MSGs (types I–IV)
- k-vector compatibility filtering with centering-aware reciprocal lattice
- Site-symmetry moment basis via SVD null-space analysis
- MAXMAGN-style maximal MSG selection
- k-SUBGROUPSMAG-style full subgroup lattice search [PerezMato2015]_
- Direct lower-symmetry MSG analysis (Bilbao/Jana2020 workflow) [Henriques2024]_
- Domain operator enumeration

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/modules

References
----------

.. [PerezMato2015] J. M. Perez-Mato, S. V. Gallego, E. S. Tasci, L. Elcoro,
   G. de la Flor, and M. I. Aroyo,
   *Annu. Rev. Mater. Res.* **45**, 217 (2015).
   `DOI:10.1146/annurev-matsci-070214-021008
   <https://doi.org/10.1146/annurev-matsci-070214-021008>`_

.. [VonDreele2024] R. B. Von Dreele and L. Elcoro,
   *Acta Cryst. B: Struct. Sci.* **80**, No. 5 (2024).
   `<https://journals.iucr.org/b/issues/2024/05/00/gar5003/index.html>`_

.. [RodriguezCarvajal2021] J. Rodriguez-Carvajal,
   *Acta Cryst. A* **77**, C176 (2021).

.. [Henriques2024] M. S. Henriques, V. Petříček, S. Goswami, and M. Dušek,
   *Acta Cryst. B: Struct. Sci.* **80**, No. 5 (2024).
   `<https://journals.iucr.org/b/issues/2024/05/00/cam5001/index.html>`_
