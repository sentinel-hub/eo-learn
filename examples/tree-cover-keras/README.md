# Tree cover prediction using deep learning

The notebooks present a toy example for training a deep learning architecture for semantic segmentation of satellite images using `eo-learn` and `keras`. The example showcases tree cover prediction over an area in France. The ground-truth data is retrieved from the [EU tree cover density (2015)](https://land.copernicus.eu/pan-european/high-resolution-layers/forests/view) through [Geopedia](http://www.geopedia.world).

## Workflow

The workflow is as follows:

  * input the area-of-interest (AOI)
  * split the AOI into small manageable eopatches
  * for each eopatch:
  * download RGB bands form Sentinel-2 L2A products using Sentinel-Hub for the 2017 year
  * retrieve corresponding ground-truth from Geopedia using a WMS request
  * compute the median values for the RGB bands over the time-interval
  * save to disk
  * select a 256x256 patch with corresponding ground-truth to be used for training/validating the model
  * train and validate a U-net

This example is presented as proof-of-concept and can easily be expanded to:

 * larger AOIs;
 * include more/different bands/indices, such as NDVI
 * include Sentinel-1 images (after harmonisation with Sentinel-2)

The notebooks require `Keras` with `tensorflow` back-end.

## Execution on AWS SageMaker

An example notebook on how to run run the workflow using [AWS SageMaker](https://aws.amazon.com/sagemaker/) is also provided. 

Instructions on how to run the notebook on SageMaker can be found [here](sagemaker.md).