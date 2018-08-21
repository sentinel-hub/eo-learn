"""
The utilities module is a collection of classes and functions used across the eolearn package, such as checking whether
two objects are deeply equal, padding of an image, etc.
"""

import dateutil.parser
import logging

LOGGER = logging.getLogger(__name__)

def parse_time(time_str):
    """ Parse input time/date string as ISO 8601 string

    :param time_str: time/date string to parse
    :type time_str: str
    :return: parsed string in ISO 8601 format
    :rtype: str
    """
    if len(time_str) < 8:
        raise ValueError('Invalid time string {}.\n'
                         'Please specify time in formats YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS'.format(time_str))
    time = dateutil.parser.parse(time_str)
    if len(time_str) <= 10:
        return time.date().isoformat()
    return time.isoformat()

def parse_time_interval(time):
    """ Parses times into common form

    Parses specified time into common form - tuple of start and end dates, i.e.:

    ``(2017-01-15:T00:00:00, 2017-01-16:T23:59:59)``

    The parameter can have the following values/format, which will be parsed as:

    * ``None`` -> `[default_start_date from config.json, current date]`
    * `YYYY-MM-DD` -> `[YYYY-MM-DD:T00:00:00, YYYY-MM-DD:T23:59:59]`
    * `YYYY-MM-DDThh:mm:ss` -> `[YYYY-MM-DDThh:mm:ss, YYYY-MM-DDThh:mm:ss]`
    * list or tuple of two dates (`YYYY-MM-DD`) -> `[YYYY-MM-DDT00:00:00, YYYY-MM-DDT23:59:59]`, where the first
      (second) element is start (end) date
    * list or tuple of two dates (`YYYY-MM-DDThh:mm:ss`) -> `[YYYY-MM-DDThh:mm:ss, YYYY-MM-DDThh:mm:ss]`,
      where the first (second) element is start (end) date

    :param time: time window of acceptable acquisitions. See above for all acceptable argument formats.
    :type time: ``None``, str of form `YYYY-MM-DD` or `'YYYY-MM-DDThh:mm:ss'`, list or tuple of two such strings
    :return: interval of start and end date of the form YYYY-MM-DDThh:mm:ss
    :rtype: tuple of start and end date
    """
    if isinstance(time, str):
        date_interval = (parse_time(time), parse_time(time))
    elif isinstance(time, list) or isinstance(time, tuple) and len(time) == 2:
        date_interval = (parse_time(time[0]), parse_time(time[1]))
    else:
        raise TabError('time must be a string or tuple of 2 strings or list of 2 strings')

    if date_interval[0] > date_interval[1]:
        raise ValueError('First time must be smaller or equal to second time')

    if len(date_interval[0].split('T')) == 1:
        date_interval = (date_interval[0] + 'T00:00:00', date_interval[1])
    if len(date_interval[1].split('T')) == 1:
        date_interval = (date_interval[0], date_interval[1] + 'T23:59:59')

    return date_interval
