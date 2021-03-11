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