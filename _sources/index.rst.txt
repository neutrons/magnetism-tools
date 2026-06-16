msgjson documentation
=====================

*msgjson* provides magnetic space group (MSG) operator tables for all 1651
commensurate Shubnikov groups and implements symmetry-analysis workflows for
magnetic structures.  The library is inspired by the k-SUBGROUPSMAG and
maximal_msgs workflows from the Bilbao Crystallographic Server [1]_, and is
designed to complement magnetic structure refinement in GSAS-II [2]_, 
FullProf [3]_, and Jana2020 [4]_.

Features
--------

- Operator tables for all 1651 commensurate MSGs (types I–IV)
- k-vector compatibility filtering with centering-aware reciprocal lattice
- Site-symmetry moment basis via SVD null-space analysis
- Maximal MSG selection (``maximal_msgs``)
- Full subgroup lattice search (``subgroup_msgs``)
- Direct lower-symmetry MSG analysis 
- Domain operator enumeration

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/modules

References
----------

.. [1] J. M. Perez-Mato, S. V. Gallego, E. S. Tasci, L. Elcoro,
   G. de la Flor, and M. I. Aroyo,
   *Annu. Rev. Mater. Res.* **45**, 217 (2015).
   `DOI:10.1146/annurev-matsci-070214-021008
   <https://doi.org/10.1146/annurev-matsci-070214-021008>`_

.. [2] R. B. Von Dreele and L. Elcoro,
   *Acta Cryst. B: Struct. Sci.* **80**, No. 5 (2024).
   `<https://journals.iucr.org/b/issues/2024/05/00/cam5001/index.html>`_

.. [3] J. Rodriguez-Carvajal, J. Gonzalez-Platas, and N. A. Katcho,
   *Acta Cryst. B: Struct. Sci.* **81**, 302 (2025).
   `<https://journals.iucr.org/b/issues/2025/03/00/cam5008/index.html>`_

.. [4] M. S. Henriques, V. Petříček, S. Goswami, and M. Dušek,
   *Acta Cryst. B: Struct. Sci.* **80**, No. 5 (2024).
   `<https://journals.iucr.org/b/issues/2024/05/00/gar5003/index.html>`_
