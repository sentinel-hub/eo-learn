## [Version 1.4.2] - 2023-3-14

- Introduced support for Python 3.11.
- Removed support for Python 3.7.
- Added T-Digest `EOTask` in the scope of the Global Earth Monitor Project, contributed by @meengel.
- Used evalscript generation utility from `sentinelhub-py` in SH related `EOTasks`.
- Deprecated the `EOPatch.merge` method and extracted it as a function.
- Deprecated the `OVERWRITE_PATCH` permission and enforcing the usage of explicit string permissions.
- Encapsulated `FeatureDict` class as `Mapping`, removed inheritance from `dict`.
- Switched to new-style typed annotations.
- Introduced the `ruff` python linter, removed `flake8` and `isort` (covered by `ruff`).
- Fixed issue with occasionally failing scheduled builds on the `master` branch.
- Various refactoring efforts and dependency improvements.
- Various improvements to tests and code.

## [Version 1.4.1] - 2023-3-14

- The codebase is now fully annotated and type annotations are mandatory for all new code.
- In the future `EOPatch` objects will **require** a valid `bbox`. For now the users are warned when no such value is provided.
- `SaveTask` and `LoadTask` now automatically save/load the bounding box whenever possible, even if not specified in `features` parameter. `CopyTask` and `MergeEOPatchesTask` also always include the bounding box when possible.
- The `EOPatch` attribute `bbox` can no longer be deleted via the `del` command.
- The `EOPatch` attribute `timestamp` was renamed into `timestamps`. The old name still works, but the users are notified. Similarly for `FeatureType.TIMESTAMP` which was renamed to `FeatureType.TIMESTAMPS`.
- Feature parsers from `eolearn.core.utils.parsers` now support callables as input for `allowed_feature_types`, which are used for filtration over all feature types. Due to this improvement the class `FeatureTypeSet` was deprecated.
- Certain rarely used methods of `FeatureType` were deprecated. Method `is_raster` has been renamed to `is_array` and designates feature types that contain numpy arrays. We also added `is_image` for types that denote temporal and timeless imagery.
- Contributors are no longer listed in file headers, but are instead listed in the `CREDITS.md` file in the root of the repository.
- Updated `CONTRIBUTING.md` instructions.
- Various other minor improvements and deprecations.


## [Version 1.4.0] - 2023-1-20

- (**codebreaking**) Complete overhaul of `eolearn.coregistration`. See documentation for details.
- (**codebreaking**) Removed non-working HVPlot backend for `eolearn.visualization`.
- (**codebreaking**) The `SpatialResizeTask` had a bug when resizing w.r.t resolution. The issue was fixed and the signature of the task was redesigned to better avoid mistakes. See documentation for details.
- (**codebreaking**) The `EOPatch` methods `get_features` and `get_feature_list` were recombined into a new `get_features` method. The method `get_time_series` was removed. See documentation for details.
- (**codebreaking**) Removed unsound `use_int_coords` option in `eolearn.ml_tools.sampling.random_point_in_triangle`.
- Added ability to specify query in execute method of `MeteoblueTask`.
- `SentinelHubInputTask` no longer saves redundant data into meta-features.
- Module `eolearn.core.utils.types` was moved to `eolearn.core.types`. Old one will be removed in the future.
- Switched `opencv-contrib-python-headless` requirement to `opencv-python-headless`
- Added type annotations to most of the code base.
- Various improvements to tests and code.


## [Version 1.3.1] - 2022-11-23

- Sentinel Hub IO tasks now support a custom timestamp filtration via `timestamp_filter` parameter, contributed by @ColinMoldenhauer.
- `MergeFeatureTask` now supports the `axis` parameter.
- Fix minor issues with the coregistration module.
- Prepare for future removal of `sentinelhub.os_utils`.
- Fix type annotations after `mypy` update.
- Improvements to tests and various minor changes.


## [Version 1.3.0] - 2022-10-06

- (**codebreaking**) Adapted Sentinel Hub tasks to `sentinelhub-py 3.8.0` which switched to Catalog 1.0.0.
- (**codebreaking**) Removed support for loading pickled objects in EOPatches (deprecated since version 1.0.0).
- (**codebreaking**) Various improvements of `FeatureIO` class. Only affects direct use of class.
- Added type annotations to majority of `eolearn.core`. The types are now exposed via `py.typed` file, which enables use of `mypy`. Added type-checking to CI for the `core` module.
- Numpy-array based features can now save and load `object` populated arrays.
- Improved documentation building, fixed links to GitHub.
- Improved test coverage.
- Added pre-commit hooks to repository for easier development.
- Various minor improvements.


## [Version 1.2.1] - 2022-09-12

- Corrected the default for `no_data_value` in `ImportFromTiffTask` and `ExportToTiffTask` to `None`. The previous default of `0` was a poor choice in many scenarios. The switch might alter behavior in existing code.
- Changed the way `SpatialResizeTask` accepts parameters for the final image size. Now supports resizing by using resolution.
- Added `ExplodeBandsTask` that explodes a multi-band feature into multiple features.
- Exposed resampling parameters in Sentinel Hub tasks and included a `geometry` execution parameter.
- Reworked internal classes `FeatureIO` and `_FeatureDict` to improve types and maintainability.
- Fixed y-axis orientation of `MeteoblueRasterTask`.
- `FilterTimeSeriesTask` adjusted to work with multiprocessing.
- EOPatch plotting no longer anti-aliases by default (removes issues with phantom values in mask plots)
- Improved documentation building, fixing a few broken links.


## [Version 1.2.0] - 2022-07-27

- Improved handling of filesystem objects:
  * introduced utility functions `pickle_fs` and `unpickle_fs` into `eo-learn-core`,
  * updated all IO tasks to fully support `filesystem` as an `__init__` parameter,
  * updated `EOExecutor` to support `filesystem` propagation to worker processes.
- Official support for Python `3.10`.
- Moved `eolearn.coregistration.ThunderRegistrationTask` into an extension of `eo-learn-coregistration` package because it doesn't support Python `3.10`.
- Updated functionality of `eolearn.features.SimpleFilterTask`. It now returns a new `EOPatch` and doesn't raise an error if all time slices would be filtered out.
- Larger updates of `eolearn.io.ExportToTiffTask`. It doesn't allow undefined `folder` parameter anymore, but it has better support for `filesystem` objects.
- Added `eolearn.core.utils.raster.fast_nanpercentile` utility function and moved `constant_pad` function into the same module.
- Suppressed a warning when saving an `EOPatch` with empty vector features.
- Updated code-style checkers by including `flake8` and checks for Jupyter notebooks.
- Various style improvements in example notebooks, code, and tests.


## [Version 1.1.1] - 2022-06-14

- Fixed a bug in `eolearn.io.ImportFromTiffTask` where a bounding box from the image wasn't added to the EOPatch.
- Increased minimal version of `Pillow` dependency in `eolearn.features`.


## [Version 1.1.0] - 2022-06-13

- Large improvements of parallelization in EOExecutor. Introduced the `eolearn.core.utils.parallelize` module, featuring tools for different parallelization modes.
- Added support for session sharing in `SentinelHubInputTask`, `SentinelHubEvalscriptTask` and `SentinelHubDemTask` by adding a `session_loader` parameter. Session sharing of `sentinelhub-py` is explained [here](https://github.com/sentinel-hub/sentinelhub-py/blob/master/examples/session_sharing.ipynb).
- Added `SpatialResizeTask` to `eolearn.features.feature_manipulation` for spatially resizing EOPatch features.
- Improved how `ImportFromTiffTask` reads from remote filesystems.
- Switched to non-structural hashing of `EONode` class to avoid massive slowdowns in large workflows.
- Improved procedure for building documentation and displaying of type annotations.
- Various minor improvements.


## [Version 1.0.2] - 2022-05-03

- Added workaround for an issue introduced by `fs==2.4.16`.
- Executor progress bar improved for use-cases with many EOPatches.
- `LoadTask` and `SaveTask` can now handle *empty* queries (by setting `eopatch_folder=None`).
- Minor improvements in code and documentation.

## [Version 1.0.1] - 2022-03-29

- Fixed an issue where vector features with empty dataframes couldn't be saved to a Geopackage.
- Memory improvement in `EOPatch` merging procedure.
- Added support for `aws_session_token`, contributed by @theirix.
- Fixed an issue in `ImportFromTiffTask`.
- Fixed a packaging issue where some new subpackage extensions didn't work in the version `1.0.0` that was released to PyPI.
- `eo-learn` abstract package from now on requires fixed versions of `eo-learn` subpackages.
- Applied `isort` formatting on the entire package.
- Minor improvements in code and documentation.

## [Version 1.0.0] - 2022-02-09

### Core Changes

- `EOPatch` changes:
  * IO for vectors and meta-info switched from `pickle` to Geopackage, GeoJSON, and JSON files. Objects saved with `pickle` can be loaded but the format is deprecated.
  * Now supports the `in` keyword for checking whether an `EOPatch` contains a given feature.
  * Major update to `EOPatch` plotting functionality, which now features a simpler `matplotlib` back-end. See [example notebook](https://github.com/sentinel-hub/eo-learn/blob/develop-v1.0/examples/visualization/EOPatchVisualization.ipynb) for more details.
  * Removed some outdated `EOPatch` methods such as `get_feature`, `rename_feature`, etc.
  * Representation (`EOPatch.__repr__` method) skips empty features.

- `EOTask` changes:
  * `EOTask` method `_parse_features` replaced with `get_feature_parser` and additional helper methods (`parse_feature`, `parse_renamed_feature`, `parse_features`, `parse_renamed_features`).
  * Removed `EOTask.__mul__` as task concatenation as it was unsound.

- `EONode` is a newly introduced object for specifying computational graphs. It replaces raw `EOTask` objects when building an `EOWorkflow`.

- `EOWorkflow` changes:
  * `LinearWorkflow` is replaced with `linearly_connect_tasks` function that prepares nodes for a linear workflow.
  * No longer accepts tuples in execution arguments. In cases where this is required, passing arguments to a task can be done with the new `InputTask`.
  * `EONodes` form a tree-like structure of dependencies, hence the end-nodes of a workflow contain all information. An `EOWorkflow` object can be constructed from end-nodes via `from_endnodes` method.

- `EOExecutor` changes:
  * Added `RayExecutor` as an extension of `EOExecutor` for working with the `ray` library.
  * Execution arguments are now given w.r.t. `EONode` objects instead of `EOTasks`.
  * Now always returns results, which by default only contain statistics. Other data (for instance the final EOPatch) can be added to results with the new `OutputTask`.
  * Additionally, supports a `filesystem` argument for saving logs and reports.
  * Reports now have the option to only link to logs, greatly reducing size in case of large numbers of EOPatches. Logs files are now also more informative.

- `FeatureParser` now supports fewer input formats but handles those better. It now returns lists instead of generators. See documentation for more information.
- `WorkflowResults` are re-done. They now contain execution stats of nodes (start and end times) and the outputs of `OutputTask`s in the workflow.
- `FeatureType` method `is_time_dependant` renamed to `is_temporal`.

### Tasks
- Added `LinearFunctionTask` which applies a linear function to features.
- `MorphologicalFilterTask` moved from `ml_tools` to `features` module.
- Sampling tasks moved `geometry` to `ml_tools` module. Sampling tasks have also been greatly upgraded, with the main being:
    - `FractionSamplingTask` for sampling random points in a class-balanced way
    - `BlockSamplingTask` for randomly sampling larger blocks of data (can also be 1 pixel blocks)
    - `GridSamplingTask` for deterministically sampling according to a grid.
- Removed `feature_extractor` module.
- Removed unused submodules of `ml_tools` (`classifier`, `postprocessing`, ...)
- To reduce core dependencies some functionalities have been moved to `extra` modules.
- Removed deprecated and outdated methods and tasks.

### Other
- Moved many examples to [new repository](https://github.com/sentinel-hub/eo-learn-examples). The rest were updated.
- Switched to github actions for CI.
- Code was reformatted with `black` and is now checked to be compliant with the standard.
- Abstract base classes are now correctly enforced.
- Added utility functions for working with S3 and AWS.
- Various minor changes.


## [Version 0.10.1] - 2021-10-27
- Copying EOPatches no longer forces loading of features if the EOPatch was loaded with `lazy_loading=True`
- `SentinelHubInputTask` now requests bands with correct units and should now work with more data collections. The parameter `bands_dtype` is now by default set to `None`, which uses the default units of each band. **Note:** due to changes the task no longer normalizes the output when `bands_dtype=np.uint16` is used.
- Minor fixes and improvements

## [Version 0.10.0] - 2021-09-14
- `EOWorkflow` now automatically makes a shallow copy of each `EOPatch` before passing it to any `EOTask` in the workflow.
- Streamlined naming conventions of EOTasks - every name now ends with `Task`. Old names have been deprecated.
- Improved functionality of merging EOPatches, particularly of merging time-dependent features.
- Removed support for Python 3.6 and added official support for Python 3.9.
- Implemented `EOPatch.copy` and `EOPatch.__delitem__` methods.
- Added `eolearn.io.MeteoblueRasterTask` and `eolearn.io.MeteoblueVectorTask` for obtaining weather data. Joint effort with Meteoblue.
- `VectorToRasterTask` now supports rasterization of time-dependant vector features. Contributed by @asylve.
- Fixes in `SentinelHubInputTask`. Both `SentinelHubInputTask` and `SentinelHubEvalscriptTask` now return EOPatches with timestamps that don't have timezone information anymore.
- Changed `rasterio` dependency to `rasterio>=1.2.7`
- All but `eolearn.core` tests ported to `pytest` framework.
- Switched from Travis CI to GitHub actions.
- Minor fixes and improvements.

## [Version 0.9.2] - 2021-05-21
- Minor fixes and improvements:
  * `SaveTask` and `LoadTask` don't automatically store a filesystem object anymore,
  * fix in `ImportFromTiff` about file extensions, contributed by @rpitonak,
  * fix in `SentinelHubInputTask` about data collection bands handling,
  * fix in `GeoDBVectorImportTask`,
  * `NormalizedDifferenceIndexTask` doesn't show division warnings anymore,
  * improvement in `PointSamplingTask`
  * improvements in LULC documentation notebook.

## [Version 0.9.1] - 2021-04-23
- Added new tasks `VectorImportTask`, `GeopediaVectorImportTask`, and `GeoDBVectorImportTask` to `eo-learn-io`.
- Code improvements in LULC classification documentation notebook.
- Minor improvements and fixes.

## [Version 0.9.0] - 2021-03-26
- Changes in `eo-learn-io` tasks that interact with Sentinel Hub services:
  * Added `SentinelHubEvalscriptTask` that downloads data given a user-defined evalscript.
  * Removed all tasks that interact with Sentinel Hub OGC services. They are fully replaced by `SentinelHubInputTask` and `SentinelHubEvalscriptTask` which use [Sentinel Hub Process API](https://docs.sentinel-hub.com/api/latest/api/process/).
  * Renamed `AddSen2CorClassificationFeature` to `SentinelHubSen2corTask`. Now it uses Process API instead of OGC.
- Changes in Sentinel-2 cloud-masking tasks:
  * Renamed `AddMultiCloudMaskTask` to `CloudMaskTask`.
  * Removed `AddCloudMaskTask` as it is superseded by `CloudMaskTask`.
  * Fixed problems with incompatibility with the latest `scikit-learn` version.
- Updated all notebooks in `eo-learn` repository.
- Minor fixes and improvements:
  * better handling of `KeyboardInterrupt` in `EOExecutor`,
  * fixed plotting of raster features with binary dtype,
  * documentation fixes.

## [Version 0.8.1] - 2021-03-05
- Added [example use-case about creating a timelapse with batch processing](https://github.com/sentinel-hub/eo-learn/tree/master/examples/batch-processing/how_to_timelapse).
- Added [example use-case about crop type classification](https://github.com/sentinel-hub/eo-learn/tree/master/examples/crop-type-classification). Contributed by @Gnilliw
- Updated code to be compatible with the latest releases `sentinelhub-py` `3.2.0` and `s2cloudless` `1.5.0`.
- Fixed an issue in `eolearn.coregistration.RegistrationTask`.
- Fixed an issue in `eolearn.io.ExportToTiff` with paths on Windows.
- Various minor improvements.

## [Version 0.8.0] - 2020-10-19
- Switched from "data source" to "data collection" terminology according to changes in `sentinelhub-py` and Sentinel Hub services.
- Improvements in `SentinelHubInputTask` to better support any type of data collection. Using new `DataCollection` class from `sentinelhub-py`.
- Extended `ExportToTiff` and `ImportFromTiff` tasks to support writing and reading from AWS S3 buckets. Implemented in cooperation with @wouellette.
- Implemented `EOPatch.merge` method and `MergeEOPatchesTask` task for merging the content of any number of EOPatches. Implemented in cooperation with @wouellette.
- Deprecated `EOPatch.concatenate` in favour of `EOPatch.merge`.
- Added `eolearn.features.DoublyLogisticApproximationTask`, contributed by @bsircelj.
- Optional parameter `config` for `SaveTask` and `LoadTask` to enable defining custom AWS credentials.
- Fixed a bug in `eolearn.features.ValueFilloutTask`.
- Started releasing `eo-learn` (sub)packages also as wheels.
- Minor improvements and fixes.

## [Version 0.7.7] - 2020-08-03
- Support for `geopandas` version `0.8.0`
- Added a [notebook](https://github.com/sentinel-hub/eo-learn/blob/develop/examples/custom-script/machine-learning-evalscript.ipynb) with an end-to-end example on how to transform a ML-model into an evalscript and run it with Sentinel Hub service
- Added `eolearn.features.ClusteringTask`, contributed by @bsircelj
- An option to define a custom log filter for `EOExecutor`
- Data mask obtained by `SentinelHubInputTask` has now boolean type instead of uint8
- Updates of some example notebooks
- A few minor fixes

## [Version 0.7.6] - 2020-07-06
- Added eo-learn dockerfiles and deployed official eo-learn docker images to [Docker Hub](https://hub.docker.com/r/sentinelhub/eolearn)
- Added `compress` parameter to `ExportToTiff`, contributed by @atedstone
- Minor fixes

## [Version 0.7.5] - 2020-06-24
- Updated example notebooks - replaced OGC service tasks with Processing API service tasks
- Deprecated tasks that download data from Sentinel Hub OGC service
- Minor fixes in `SentinelHubInputTask` and `AddCloudMaskTask`

## [Version 0.7.4] - 2020-05-14
- Updates of `SentinelHubInputTask`:
  * Support for new s2cloudless precomputed cloud mask ([more info](https://medium.com/sentinel-hub/cloud-masks-at-your-service-6e5b2cb2ce8a))
  * Support for `config` parameter
- Updated `SI_LULC_pipeline` notebook.

## [Version 0.7.3] - 2020-03-16
- Added support for `geopandas` version `0.7.0`.
- Fixed a bug in `eolearn.core.eodata_io.save_eopatch` function.
- Improvement in `eolearn.mask.MaskFeature` task - it now works also works with time independent feature types.
- A minor improvement in `eolearn.io.SentinelHubInputTask` task.

## [Version 0.7.2] - 2020-02-17
- Support additional data in the Processing API input task (such as sunAzimuthAngles, sunZenithAngles, viewAzimuthMean, viewZenithMean)
- Compatibility with the `sentinelhub-py` 3.0
- Removed support for python 3.5
- Multiprocessing Log filtering

## [Version 0.7.1] - 2020-02-05
### Fixed
- `eolearn.io.SentinelHubInputTask`: evalscript version was not passed to the sentinel-hub service.
- `eolearn.core.EOWorkflow`: fixed generating task dependencies.
### Added
- Processing API docs generation.
- Introduced CHANGELOG.md.
