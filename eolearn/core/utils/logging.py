"""
The utilities module is a collection of classes and functions used across the eolearn package, such as checking whether
two objects are deeply equal, padding of an image, etc.

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from logging import Filter, LogRecord
from typing import Any

LOGGER = logging.getLogger(__name__)


class LogFileFilter(Filter):
    """Filters log messages passed to log file."""

    def __init__(self, thread_name: str | None, *args: Any, **kwargs: Any):
        """
        :param thread_name: Name of the thread by which to filter logs. By default, it won't filter by any name.
        """
        self.thread_name = thread_name
        super().__init__(*args, **kwargs)

    def filter(self, record: LogRecord) -> bool:
        """Shows everything from the thread that it was initialized in."""
        return record.threadName == self.thread_name
