# Land Use and Land Cover (LULC) classification with [Batch Processing](https://docs.sentinel-hub.com/api/latest/api/batch/)

This example shows how to replace part of the [Land-User-Land-Cover Prediction](https://eo-learn.readthedocs.io/en/latest/examples/land-cover-map/SI_LULC_pipeline.html)  `eo-learn` workflow with Batch Processing to optimise the processing of large areas. The acquisition of the satellite data, processing of derived products and resampling over a uniform time-range is performed with Sentinel Hub services. The approach allows for much faster processing times over large areas, reduced costs and less computational resources to process the data.



## Overview

This notebook is organised in 2 main sections:

- **Part 1** is dedicated to creating and running the Batch Process.
- **Part 2** focuses on converting the results obtained in *Part 1* to [EOPatches](https://eo-learn.readthedocs.io/en/latest/examples/core/CoreOverview.html#EOPatch), the format used in [EOLearn](https://eo-learn.readthedocs.io/en/latest/index.html).
- **Part 3** shows how to integrate the workflow into the [LULC pipeline](https://github.com/sentinel-hub/eo-learn/blob/master/examples/land-cover-map/SI_LULC_pipeline.ipynb) to predict LULC using machine learning algorithms.