# Poverty Detection from Sentinel-2 and VIIRS

This example shows how three data collections -- Sentinel-2A, VIIRS night lights, and the USAID Demographic and Health Survey -- can be used to predict neighborhood poverty levels in Rwanda.

## Additional information

In addition to installing the `eo-learn` library, this example requires four other libraries to be installed:
- [`pillow`](https://pillow.readthedocs.io/en/stable/)
- [`rasterstats`](https://pythonhosted.org/rasterstats/)
- [`keras`](https://keras.io/)
- [`hyperopt`](http://hyperopt.github.io/hyperopt/)

To keep the notebook (more) concise, configuration, model definition, and utility functions are defined in standalone python files (`pred_cong_pd.py`, `train_config_pd.py`, `utils.py`, `Vgg16.py`).
