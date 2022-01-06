"""
Implementation of custom eo-learn exceptions and warnings
"""
import warnings


class EODeprecationWarning(DeprecationWarning):
    """ A custom deprecation warning for eo-learn package
    """


class EOUserWarning(UserWarning):
    """ A custom user warning for eo-learn package
    """


class EORuntimeWarning(RuntimeWarning):
    """ A custom runtime warning for eo-learn package
    """


warnings.simplefilter('default', EODeprecationWarning)
warnings.simplefilter('default', EOUserWarning)
warnings.simplefilter('always', EORuntimeWarning)
