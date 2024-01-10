"""Main module of the `eolearn` package."""

__version__ = "1.5.3"

import importlib.util
import warnings

SUBPACKAGES = ["core", "coregistration", "features", "geometry", "io", "mask", "ml_tools", "visualization"]
deprecated_installs = [
    subpackage for subpackage in SUBPACKAGES if importlib.util.find_spec(f"deprecated_eolearn_{subpackage}") is not None
]
if deprecated_installs:
    warnings.warn(
        f"You are currently using an outdated installation of `eo-learn` for submodules {deprecated_installs}. You can"
        " find instructions on how to install `eo-learn` correctly at"
        " https://github.com/sentinel-hub/eo-learn/issues/733."
    )
