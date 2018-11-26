[![Package version](https://badge.fury.io/py/eo-learn.svg)](https://pypi.org/project/eo-learn/)
[![Build Status](https://travis-ci.org/sentinel-hub/eo-learn.svg?branch=master)](https://travis-ci.org/sentinel-hub/eo-learn)
[![Docs status](https://readthedocs.org/projects/eo-learn/badge/?version=latest)](https://eo-learn.readthedocs.io)
[![License](https://img.shields.io/pypi/l/eo-learn.svg)](https://github.com/sentinel-hub/eo-learn/blob/master/LICENSE)
<img align="right" src="docs/source/figures/eo-learn-logo.png" alt="drawing" width="300"/>


# eo-learn
**`eo-learn` makes extraction of valuable information from satellite imagery easy.**

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

![eo-learn-workflow0illustration](docs/source/figures/eo-learn-illustration.png)

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

## Installation

The package requires Python version `>=3.5`. It can be installed with:

```bash
pip install eo-learn
```

however it is also possible to install each subpackage separately:

```bash
pip install eo-learn-core
pip install eo-learn-coregistration
pip install eo-learn-features
pip install eo-learn-geometry
pip install eo-learn-io
pip install eo-learn-mask
pip install eo-learn-ml-tools
```

## Documentation

For more information on the package content, visit [readthedocs](https://eo-learn.readthedocs.io/).


# Blog posts

 * [Introducing eo-learn](https://medium.com/sentinel-hub/introducing-eo-learn-ab37f2869f5c)
 * [Land Cover Classification with eo-learn: Part 1 - Mastering Satellite Image Data in an Open-Source Python Environment](https://medium.com/sentinel-hub/land-cover-classification-with-eo-learn-part-1-2471e8098195)


## License

See [LICENSE](LICENSE).
