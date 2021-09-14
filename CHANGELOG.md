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