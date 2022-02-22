"""
Credits:
Copyright (c) 2021-2022 Matej Aleksandrov, Matej Batič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import pytest

from sentinelhub import BBox

from eolearn.core import FeatureType
from eolearn.io.extra.geodb import GeoDBVectorImportTask


@pytest.mark.parametrize(
    argnames="reproject, clip, n_features, bbox",
    ids=["bbox", "bbox_smaller"],
    argvalues=[
        (False, False, 227, BBox([857000, 6521500, 861000, 6525500], crs="epsg:2154")),
        (True, True, 162, BBox([857400, 6521900, 860600, 6525100], crs="epsg:2154")),
    ],
)
def test_import_from_geodb(geodb_client, reproject, clip, n_features, bbox):
    """Test for importing from GeoDB.
    It will only run if geodb credentials are available as the environment variables
    """
    assert geodb_client.whoami, "Client is not set-up correctly"

    feature = FeatureType.VECTOR_TIMELESS, "lpis_iacs"
    import_task = GeoDBVectorImportTask(
        feature=feature,
        geodb_client=geodb_client,
        geodb_collection="france_metropolitan_gsaa_2015",
        geodb_db="lpis_iacs",
        reproject=reproject,
        clip=clip,
    )
    eop = import_task.execute(bbox=bbox)
    assert len(eop[feature]) == n_features, "Wrong number of features!"
    assert eop[feature].crs.to_epsg() == eop.bbox.crs.epsg
