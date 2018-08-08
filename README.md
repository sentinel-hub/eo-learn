[![Build Status](https://travis-ci.org/sentinel-hub/eo-learn.svg?branch=master)](https://travis-ci.org/sentinel-hub/eo-learn)
[![Docs status](https://readthedocs.org/projects/eo-learn/badge/?version=latest)](https://eo-learn.readthedocs.io)

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

Each of the subpackages can be installed separately

```bash
pip install git+https://github.com/sentinel-hub/eo-learn#subdirectory=core --upgrade
pip install git+https://github.com/sentinel-hub/eo-learn#subdirectory=coregistration --upgrade
pip install git+https://github.com/sentinel-hub/eo-learn#subdirectory=features --upgrade
pip install git+https://github.com/sentinel-hub/eo-learn#subdirectory=geometry --upgrade
pip install git+https://github.com/sentinel-hub/eo-learn#subdirectory=io --upgrade
pip install git+https://github.com/sentinel-hub/eo-learn#subdirectory=mask --upgrade
pip install git+https://github.com/sentinel-hub/eo-learn#subdirectory=ml_tools --upgrade
```

To install all packages at once you can download the repository and call

```bash
python install_all.py
```

## Documentation

For more information on the package content, visit [readthedocs](https://eo-learn.readthedocs.io/).

## License

See [LICENSE](LICENSE).
