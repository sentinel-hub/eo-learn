"""
Implementation of custom eo-learn exceptions and warnings

Credits:
Copyright (c) 2021-2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import warnings


class EODeprecationWarning(DeprecationWarning):
    """A custom deprecation warning for eo-learn package."""


class EOUserWarning(UserWarning):
    """A custom user warning for eo-learn package."""


class EORuntimeWarning(RuntimeWarning):
    """A custom runtime warning for eo-learn package."""


warnings.simplefilter("default", EODeprecationWarning)
warnings.simplefilter("default", EOUserWarning)
warnings.simplefilter("always", EORuntimeWarning)


def renamed_and_deprecated(deprecated_class):
    """A class decorator that signals that the class has been renamed when initialized.

    Example of use:

    .. code-block:: python

        @renamed_and_deprecated
        class OldNameForClass(NewNameForClass):
            ''' Deprecated version of `NewNameForClass`
            '''

    """

    def warn_and_init(self, *args, **kwargs):
        warnings.warn(
            f"The class {self.__class__.__name__} has been renamed to {self.__class__.__mro__[1].__name__}. "
            "The old name is deprecated and will be removed in version 1.0",
            EODeprecationWarning,
        )
        super(deprecated_class, self).__init__(*args, **kwargs)

    deprecated_class.__init__ = warn_and_init
    return deprecated_class
