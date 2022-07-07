"""
The utilities module is a collection of classes and functions used across the eolearn package, such as checking whether
two objects are deeply equal, padding of an image, etc.

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Grega Milčinski, Domagoj Korais, Matic Lubej (Sinergise)
Copyright (c) 2017-2022 Žiga Lukšič, Devis Peressutti, Tomislav Slijepčević, Nejc Vesel, Jovan Višnjić (Sinergise)
Copyright (c) 2017-2022 Anže Zupanc (Sinergise)
Copyright (c) 2019-2020 Jernej Puc, Lojze Žust (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Andrej Burja (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import logging
from logging import Filter, LogRecord
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)


class LogFileFilter(Filter):
    """Filters log messages passed to log file."""

    def __init__(self, thread_name: Optional[str], *args: Any, **kwargs: Any):
        """
        :param thread_name: Name of the thread by which to filter logs. By default, it won't filter by any name.
        """
        self.thread_name = thread_name
        super().__init__(*args, **kwargs)

    def filter(self, record: LogRecord) -> bool:
        """Shows everything from the thread that it was initialized in."""
        return record.threadName == self.thread_name
