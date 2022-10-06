"""
A module implementing utilities for working with geopackage files

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
# pylint: skip-file
import numpy as np
from geopandas.io.file import _geometry_types


def infer_schema(df):
    """This function is copied over from GeoPandas, part of geopandas.io.file module, with the sole purpose of
    disabling the `if df.empty` check that prevents saving an empty dataframe. Will be removed after GeoPandas
    version 0.11 fixes the problem.
    """

    from collections import OrderedDict

    # TODO: test pandas string type and boolean type once released
    types = {"Int64": "int", "string": "str", "boolean": "bool"}

    def convert_type(column, in_type):
        if in_type == object:
            return "str"
        if in_type.name.startswith("datetime64"):
            # numpy datetime type regardless of frequency
            return "datetime"
        if str(in_type) in types:  # noqa
            out_type = types[str(in_type)]
        else:
            out_type = type(np.zeros(1, in_type).item()).__name__
        if out_type == "long":
            out_type = "int"
        return out_type

    properties = OrderedDict(
        [
            (col, convert_type(col, _type))
            for col, _type in zip(df.columns, df.dtypes)
            if col != df._geometry_column_name
        ]
    )

    # NOTE: commented out by eo-learn team
    # if df.empty:
    #     raise ValueError("Cannot write empty DataFrame to file.")

    # Since https://github.com/Toblerity/Fiona/issues/446 resolution,
    # Fiona allows a list of geometry types
    geom_types = _geometry_types(df)

    schema = {"geometry": geom_types, "properties": properties}

    return schema
