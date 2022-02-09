[![Package version](https://badge.fury.io/py/eo-learn.svg)](https://pypi.org/project/eo-learn)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/eo-learn.svg)](https://anaconda.org/conda-forge/eo-learn)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/eo-learn.svg?style=flat-square)](https://pypi.org/project/eo-learn)
[![Build Status](https://github.com/sentinel-hub/eo-learn/actions/workflows/ci_action.yml/badge.svg?branch=master)](https://github.com/sentinel-hub/eo-learn/actions)
[![Docs status](https://readthedocs.org/projects/eo-learn/badge/?version=latest)](https://eo-learn.readthedocs.io)
[![License](https://img.shields.io/pypi/l/eo-learn.svg)](https://github.com/sentinel-hub/eo-learn/blob/master/LICENSE)
[![Overall downloads](http://pepy.tech/badge/eo-learn)](https://pepy.tech/project/eo-learn)
[![Last month downloads](https://pepy.tech/badge/eo-learn/month)](https://pepy.tech/project/eo-learn)
[![Docker pulls](https://img.shields.io/docker/pulls/sentinelhub/eolearn.svg)](https://hub.docker.com/r/sentinelhub/eolearn)
[![Code coverage](https://codecov.io/gh/sentinel-hub/eo-learn/branch/master/graph/badge.svg)](https://codecov.io/gh/sentinel-hub/eo-learn)
<img align="right" src="docs/source/figures/eo-learn-logo.png" alt="" width="300"/>


# eo-learn
**eo-learn makes extraction of valuable information from satellite imagery easy.**

The availability of open Earth observation (EO) data through the Copernicus and Landsat programs represents an
unprecedented resource for many EO applications, ranging from ocean and land use and land cover monitoring,
disaster control, emergency services and humanitarian relief. Given the large amount of high spatial resolution
data at high revisit frequency, techniques able to automatically extract complex patterns in such _spatio-temporal_
data are needed.

**`eo-learn`** is a collection of open source Python packages that have been developed to seamlessly access and process
_spatio-temporal_ image sequences acquired by any satellite fleet in a timely and automatic manner. **`eo-learn`** is
easy to use, it's design modular, and encourages collaboration -- sharing and reusing of specific tasks in a typical
EO-value-extraction workflows, such as cloud masking, image co-registration, feature extraction, classification, etc. Everyone is free
to use any of the available tasks and is encouraged to improve the, develop new ones and share them with the rest of the community.

**`eo-learn`** makes extraction of valuable information from satellite imagery as easy as defining a sequence of operations to be performed on satellite imagery. Image below illustrates a processing chain that maps water in satellite imagery by thresholding the Normalised Difference Water Index in user specified region of interest.

![](docs/source/figures/eo-learn-illustration.png)

**`eo-learn`** _library acts as a bridge between Earth observation/Remote sensing field and Python ecosystem for data science and machine learning._ The library is written in Python and uses NumPy arrays to store and handle remote sensing data. Its aim is to make entry easier for non-experts to the field of remote sensing on one hand and bring the state-of-the-art tools for computer vision, machine learning, and deep learning existing in Python ecosystem to remote sensing experts.

## Package Overview

**`eo-learn`** is divided into several subpackages according to different functionalities and external package dependencies. Therefore it is not necessary for user to install entire package but only the parts that he needs.

At the moment there are the following subpackages:

- **`eo-learn-core`** - The main subpackage which implements basic building blocks (`EOPatch`, `EOTask` and `EOWorkflow`) and commonly used functionalities.
- **`eo-learn-coregistration`** - The subpackage that deals with image co-registraion.
- **`eo-learn-features`** - A collection of utilities for extracting data properties and feature manipulation.
- **`eo-learn-geometry`** - Geometry subpackage used for geometric transformation and conversion between vector and raster data.
- **`eo-learn-io`** - Input/output subpackage that deals with obtaining data from Sentinel Hub services or saving and loading data locally.
- **`eo-learn-mask`** - The subpackage used for masking of data and calculation of cloud masks.
- **`eo-learn-ml-tools`** - Various tools that can be used before or after the machine learning process.
- **`eo-learn-visualization`** - Visualization tools for core elements of eo-learn.

## Installation

### PyPi distribution

The package requires Python version **>=3.7** . It can be installed with:

```bash
pip install eo-learn
```

In order to avoid heavy package dependencies it is possible to install each subpackage separately:
```bash
pip install eo-learn-core
pip install eo-learn-coregistration
pip install eo-learn-features
pip install eo-learn-geometry
pip install eo-learn-io
pip install eo-learn-mask
pip install eo-learn-ml-tools
pip install eo-learn-visualization
```

Before installing `eo-learn` on **Windows** it is recommended to install the following packages from [Unofficial Windows wheels repository](https://www.lfd.uci.edu/~gohlke/pythonlibs/):

```bash
gdal
rasterio
shapely
fiona
```

One of dependencies of `eo-learn-mask` subpackage is `lightgbm` package. On Windows it requires 64 bit Python distribution. If having problems during installation please check [LightGBM installation guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html).

Some subpackages contain extension modules under `extra` subfolder. Those modules typically require additional package dependencies which don't get installed by default.

### Conda Forge distribution

The package requires a Python environment **>=3.7**. 

Thanks to the maintainers of the conda forge feedstock (@benhuff, @dcunn, @mwilson8, @oblute, @rluria14), `eo-learn` can 
be installed using `conda-forge` as follows:

```bash
conda config --add channels conda-forge

conda install eo-learn
```

In order to avoid heavy package dependencies it is possible to install each subpackage separately:

```bash
conda install eo-learn-core
conda install eo-learn-coregistration
conda install eo-learn-features
conda install eo-learn-geometry
conda install eo-learn-io
conda install eo-learn-mask
conda install eo-learn-ml-tools
conda install eo-learn-visualization
```

### Run with Docker

A docker image with the latest released version of `eo-learn` is available at [Docker Hub](https://hub.docker.com/r/sentinelhub/eolearn). It provides a full installation of `eo-learn` together with a Jupyter notebook environment. You can pull and run it with:

```bash
docker pull sentinelhub/eolearn:latest
docker run -p 8888:8888 sentinelhub/eolearn:latest
```

An extended version of the `latest` image additionally contains all example notebooks and data to get you started with `eo-learn`. Run it with: 

```bash
docker pull sentinelhub/eolearn:latest-examples
docker run -p 8888:8888 sentinelhub/eolearn:latest-examples
```

Both docker images can also be built manually from GitHub repository:

```bash
docker build -f docker/eolearn.dockerfile . --tag=sentinelhub/eolearn:latest
docker build -f docker/eolearn-examples.dockerfile . --tag=sentinelhub/eolearn:latest-examples
```


## Documentation

For more information on the package content, visit [readthedocs](https://eo-learn.readthedocs.io/).

## More Examples

Examples and introductions to the package can be found [here](https://github.com/sentinel-hub/eo-learn/tree/master/examples). A large collection of examples is available at the [`eo-learn-examples`](https://github.com/sentinel-hub/eo-learn-examples) repository. While the examples there are not always up-to-date they can be a great source of ideas.

## Contributions

If you would like to contribute to `eo-learn`, check out our [contribution guidelines](./CONTRIBUTING.md).

## Blog posts and papers

 * [Introducing eo-learn](https://medium.com/sentinel-hub/introducing-eo-learn-ab37f2869f5c) (by Devis Peressutti)
 * [Land Cover Classification with eo-learn: Part 1 - Mastering Satellite Image Data in an Open-Source Python Environment](https://medium.com/sentinel-hub/land-cover-classification-with-eo-learn-part-1-2471e8098195) (by Matic Lubej)
 * [Land Cover Classification with eo-learn: Part 2 - Going from Data to Predictions in the Comfort of Your Laptop](https://medium.com/sentinel-hub/land-cover-classification-with-eo-learn-part-2-bd9aa86f8500) (by Matic Lubej)
 * [Land Cover Classification with eo-learn: Part 3 - Pushing Beyond the Point of “Good Enough”](https://medium.com/sentinel-hub/land-cover-classification-with-eo-learn-part-3-c62ed9ecd405) (by Matic Lubej)
 * [Innovations in satellite measurements for development](https://blogs.worldbank.org/opendata/innovations-satellite-measurements-development)
 * [Use eo-learn with AWS SageMaker](https://medium.com/@drewbo19/use-eo-learn-with-aws-sagemaker-9420856aafb5) (by Drew Bollinger)
 * [Spatio-Temporal Deep Learning: An Application to Land Cover Classification](https://www.researchgate.net/publication/333262625_Spatio-Temporal_Deep_Learning_An_Application_to_Land_Cover_Classification) (by Anze Zupanc)
 * [Tree Cover Prediction with Deep Learning](https://medium.com/dataseries/tree-cover-prediction-with-deep-learning-afeb0b663966) (by Daniel Moraite)
 * [NoRSC19 Workshop on eo-learn](https://github.com/sentinel-hub/norsc19-eo-learn-workshop)
 * [Tracking a rapidly changing planet](https://medium.com/@developmentseed/tracking-a-rapidly-changing-planet-bc02efe3545d) (by Development Seed)
 * [Land Cover Monitoring System](https://medium.com/sentinel-hub/land-cover-monitoring-system-84406e3019ae) (by Jovan Visnjic and Matej Aleksandrov)
 * [eo-learn Webinar](https://www.youtube.com/watch?v=Rv-yK7Vbk4o) (by Anze Zupanc)
 * [Cloud Masks at Your Service](https://medium.com/sentinel-hub/cloud-masks-at-your-service-6e5b2cb2ce8a) 
 * [ML examples for Common Agriculture Policy](https://medium.com/sentinel-hub/area-monitoring-concept-effc2c262583) 
   * [High-Level Concept](https://medium.com/sentinel-hub/area-monitoring-concept-effc2c262583)
   * [Data Handling](https://medium.com/sentinel-hub/area-monitoring-data-handling-c255b215364f)
   * [Outlier detection](https://medium.com/sentinel-hub/area-monitoring-observation-outlier-detection-34f86b7cc63)
   * [Similarity Score](https://medium.com/sentinel-hub/area-monitoring-similarity-score-72e5cbfb33b6)
   * [Bare Soil Marker](https://medium.com/sentinel-hub/area-monitoring-bare-soil-marker-608bc95712ae)
   * [Mowing Marker](https://medium.com/sentinel-hub/area-monitoring-mowing-marker-e99cff0c2d08)
   * [Crop Type Marker](https://medium.com/sentinel-hub/area-monitoring-crop-type-marker-1e70f672bf44)
   * [Homogeneity Marker](https://medium.com/sentinel-hub/area-monitoring-homogeneity-marker-742047b834dc)
   * [Parcel Boundary Detection](https://medium.com/sentinel-hub/parcel-boundary-detection-for-cap-2a316a77d2f6)
   * Land Cover Classification (still to come)
   * Minimum Agriculture Activity (still to come)
   * [Combining the Markers into Decisions](https://medium.com/sentinel-hub/area-monitoring-combining-markers-into-decisions-d74f70fe7721)
   * Traffic Light System (still to come)
   * Expert Judgement Application (still to come)
 * [Scale-up your eo-learn workflow using Batch Processing API](https://medium.com/sentinel-hub/scale-up-your-eo-learn-workflow-using-batch-processing-api-d183b70ea237) (by Maxim Lamare) 
 

## Questions and Issues

Feel free to ask questions about the package and its use cases at [Sentinel Hub forum](https://forum.sentinel-hub.com/) or raise an issue on [GitHub](https://github.com/sentinel-hub/eo-learn/issues).

You are welcome to send your feedback to the package authors, EO Research team, through any of [Sentinel Hub communication channel](https://sentinel-hub.com/develop/communication-channels).


## License

See [LICENSE](https://github.com/sentinel-hub/eo-learn/blob/master/LICENSE).

## Acknowledgements

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreements No. 776115 and No. 101004112.
