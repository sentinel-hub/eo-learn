## [Version 0.7.2] - 2019-02-17
- Support additional data in the Processing API input task (such as sunAzimuthAngles, sunZenithAngles, viewAzimuthMean, viewZenithMean)
- Compatibility with the `sentinelhub-py` 3.0
- Removed support for python 3.5
- Multiprocessing Log filtering

## [Version 0.7.1] - 2019-02-05
### Fixed
- `eolearn.io.SentinelHubInputTask`: evalscript version was not passed to the sentinel-hub service.
- `eolearn.core.EOWorkflow`: fixed generating task dependencies.
### Added
- Processing API docs generation.
- Introduced CHANGELOG.md.