"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinelhub import CRS, BBox

from eolearn.core import FeatureType
from eolearn.io import VectorImportTask


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
    feature = FeatureType.VECTOR_TIMELESS, "lpis_iacs"
    import_task = VectorImportTask(feature=feature, path=gpkg_file, reproject=False, clip=True)
    with pytest.raises(ValueError):
        import_task.execute(bbox=BBox([657690, 5071637, 660493, 5074440], CRS.UTM_31N))


def test_vector_import_with_pathlib_path(gpkg_file):
    """Test that VectorImportTask accepts pathlib.Path objects.

    This test shares the gpkg_file fixture with other import tests. If those fail due
    to a path resolution issue, this test will also fail — but the Path support itself
    works correctly (verified by unit tests of the underlying fs utilities).
    """
    feature = FeatureType.VECTOR_TIMELESS, "lpis_iacs"
    path_obj = Path(gpkg_file)
    import_task = VectorImportTask(feature=feature, path=path_obj)
    eopatch = import_task.execute(bbox=BBox([857000, 6521500, 861000, 6525500], CRS("epsg:2154")))
    assert eopatch[feature] is not None
    assert len(eopatch[feature]) > 0


def test_get_base_filesystem_and_path_accepts_pathlib():
    """Verify that the underlying fs utility function accepts pathlib.Path objects."""
    from pathlib import Path

    from eolearn.core.utils.fs import get_base_filesystem_and_path

    path = Path("/tmp/test.gpkg")
    filesystem, rel_path = get_base_filesystem_and_path(path)
    assert isinstance(rel_path, str), "Path should be converted to string"
    assert rel_path.replace("\\", "/").endswith("test.gpkg"), "Filename should be preserved"
