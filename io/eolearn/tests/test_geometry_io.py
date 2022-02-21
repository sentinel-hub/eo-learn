"""
Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
import pytest

from sentinelhub import CRS, BBox

from eolearn.core import FeatureType
from eolearn.io import GeopediaVectorImportTask, VectorImportTask


@pytest.mark.parametrize(
    argnames="reproject, clip, n_features, bbox, crs",
    ids=["simple", "bbox", "bbox_full", "bbox_smaller"],
    argvalues=[
        (False, False, 193, None, None),
        (False, False, 193, BBox([857000, 6521500, 861000, 6525500], CRS("epsg:2154")), None),
        (True, True, 193, BBox([657089, 5071037, 661093, 5075039], CRS.UTM_31N), CRS.UTM_31N),
        (True, True, 125, BBox([657690, 5071637, 660493, 5074440], CRS.UTM_31N), CRS.UTM_31N),
    ],
)
class TestVectorImportTask:
    """Class for testing vector imports from local file, s3 bucket object and layer from Geopedia"""

    def test_import_local_file(self, gpkg_file, reproject, clip, n_features, bbox, crs):
        self._test_import(bbox, clip, crs, gpkg_file, n_features, reproject)

    def test_import_s3_file(self, s3_gpkg_file, reproject, clip, n_features, bbox, crs):
        self._test_import(bbox, clip, crs, s3_gpkg_file, n_features, reproject)

    def test_import_from_geopedia(self, reproject, clip, n_features, bbox, crs):
        feature = FeatureType.VECTOR_TIMELESS, "lpis_iacs"
        import_task = GeopediaVectorImportTask(feature=feature, geopedia_table=3447, reproject=reproject, clip=clip)
        eop = import_task.execute(bbox=bbox)
        assert len(eop[feature]) == n_features, "Wrong number of features!"
        to_crs = crs or import_task.dataset_crs
        assert eop[feature].crs.to_epsg() == to_crs.epsg

    @staticmethod
    def _test_import(bbox, clip, crs, gpkg_example, n_features, reproject):
        feature = FeatureType.VECTOR_TIMELESS, "lpis_iacs"
        import_task = VectorImportTask(feature=feature, path=gpkg_example, reproject=reproject, clip=clip)
        eop = import_task.execute(bbox=bbox)
        assert len(eop[feature]) == n_features, "Wrong number of features!"
        to_crs = crs or import_task.dataset_crs
        assert eop[feature].crs == to_crs.pyproj_crs()


def test_clipping_wrong_crs(gpkg_file):
    """Test for trying to clip using different CRS than the data is in"""
    with pytest.raises(ValueError):
        feature = FeatureType.VECTOR_TIMELESS, "lpis_iacs"
        import_task = VectorImportTask(feature=feature, path=gpkg_file, reproject=False, clip=True)
        import_task.execute(bbox=BBox([657690, 5071637, 660493, 5074440], CRS.UTM_31N))
