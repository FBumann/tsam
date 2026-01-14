################
tsam's Change Log
################

*********************
Release version 3.0.0
*********************

tsam v3.0.0 is a major release introducing a modern, functional API alongside significant improvements to plotting, hyperparameter tuning, and overall code quality.

Breaking Changes
================

* **New functional API**: The primary interface is now ``tsam.aggregate()`` which returns an ``AggregationResult`` object
* **Configuration objects**: Clustering and segmentation options are now configured via ``ClusterConfig`` and ``SegmentConfig`` dataclasses
* **Renamed parameters**: Many parameters have been renamed for consistency (e.g., ``noTypicalPeriods`` → ``n_clusters``, ``hoursPerPeriod`` → ``period_duration``)
* **Removed methods**: The ``reconstruct()`` method has been removed; use the ``reconstructed`` property on results instead

New Features
============

* **Modern functional API**: New ``tsam.aggregate()`` function with clear configuration objects
* **Rich result objects**: ``AggregationResult`` provides easy access to cluster representatives, accuracy metrics, and reconstructed data
* **Clustering transfer**: New ``ClusteringResult`` dataclass enables serialization and transfer of clustering configurations
* **Improved plotting**: New plotting methods with better defaults and Plotly integration
* **Enhanced hyperparameter tuning**: Improved ``find_optimal_combination()`` and ``find_pareto_front()`` with additional parameters:

  - ``segment_representation``
  - ``extremes`` configuration
  - ``preserve_column_means``
  - ``round_decimals``
  - ``numerical_tolerance``

* **Accuracy metrics**: New ``AccuracyMetrics`` class with convenient summary methods

Improvements
============

* Segment center preservation for better accuracy
* Consistent semantic naming across the entire codebase
* Better handling of extreme periods with ``n_clusters`` edge cases
* Fixed rescaling with segmentation (was applying rescaling twice)
* Fixed ``predictOriginalData()`` denormalization when using ``sameMean=True`` with segmentation

Legacy API
==========

The class-based API remains available for backward compatibility::

    import tsam.timeseriesaggregation as tsam_legacy
    aggregation = tsam_legacy.TimeSeriesAggregation(raw, noTypicalPeriods=8, ...)

See the migration examples in the documentation for upgrading to the new API.


*********************
Release version 2.3.9
*********************

* Improved time series aggregation speed with segmentation (issue #96)
* Fixed issue #99


*********************
Release version 2.3.8
*********************

* Enhanced time series aggregation speed with segmentation (issue #96)


*********************
Release version 2.3.7
*********************

* Added Python 3.13 support
* Updated GitHub Actions workflow (ubuntu-20.04 → ubuntu-22.04)
* Resolved invalid escape sequence error (issue #90)


*********************
Release version 2.3.6
*********************

* Migrated from ``setup.py`` to ``pyproject.toml``
* Changed project layout from flat to source structure
* Updated installation documentation
* Fixed deprecation and future warnings (issue #91)


*********************
Release version 2.3.5
*********************

* Re-release of v2.3.4 to fix GitHub/PyPI synchronization


*********************
Release version 2.3.4
*********************

* Extended reporting for time series tolerance exceedances
* Added option to silence tolerance warnings (default threshold: 1e-13)


*********************
Release version 2.3.3
*********************

* Dropped support for Python versions below 3.9
* Fixed deprecation warnings


*********************
Release version 2.3.2
*********************

* Limited pandas version to below 3.0
* Silenced deprecation warnings


*********************
Release version 2.3.1
*********************

* Accelerated rescale cluster periods functionality
* Updated documentation with autodeployment features


*********************
Release version 2.3.0
*********************

* Fixed deprecated pandas functions
* Corrected distribution representation sum calculations
* Added segment representation capability
* Extended default example
* Switched CI infrastructure from Travis to GitHub workflows


*********************
Release version 2.2.2
*********************

* Fixed Hypertuning class
* Adjusted the default MILP solver
* Reworked documentation


*********************
Release version 2.1.0
*********************

* Added hyperparameter tuning meta class for identifying optimal time series aggregation parameters


*********************
Release version 2.0.1
*********************

* Changed dependency of scikit-learn to make tsam conda-forge compatible


*********************
Release version 2.0.0
*********************

* A new comprehensive structure that allows for free cross-combination of clustering algorithms and cluster representations (e.g., centroids or medoids)
* A novel cluster representation method that precisely replicates the original time series value distribution based on `Hoffmann, Kotzur and Stolten (2021) <https://arxiv.org/abs/2111.12072>`_
* Maxoids as representation algorithm which represents time series by outliers only based on Sifa and Bauckhage (2017): "Online k-Maxoids clustering"
* K-medoids contiguity: An algorithm based on Oehrlein and Hauner (2017) that accounts for contiguity constraints


*********************
Release version 1.1.2
*********************

* Added first version of the k-medoid contiguity algorithm


*********************
Release version 1.1.1
*********************

* Significantly increased test coverage
* Separation between clustering and representation (e.g., for Ward's hierarchical clustering, the representation by medoids or centroids can now be freely chosen)


*********************
Release version 1.1.0
*********************

* Segmentation (clustering of adjacent time steps) according to Pineda et al. (2018)
* k-MILP: Extension of MILP-based k-medoids clustering for automatic identification of extreme periods according to Zatti et al. (2019)
* Option to dynamically choose whether clusters should be represented by their centroid or medoid
