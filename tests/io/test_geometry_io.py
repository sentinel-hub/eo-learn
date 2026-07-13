"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import tempfile

import geopandas as gpd
import pytest
from shapely import Point

from sentinelhub import CRS, BBox

from eolearn.core import EOPatch, FeatureType
from eolearn.io import VectorExportTask, VectorImportTask


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


def _create_test_geodataframe() -> gpd.GeoDataFrame:
    """Create a simple GeoDataFrame with a few points for testing."""
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3], "label": ["a", "b", "c"]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )


class TestVectorExportTask:
    """Tests for the VectorExportTask."""

    def test_export_gpkg(self):
        """Test exporting a vector feature to GPKG format."""
        gdf = _create_test_geodataframe()
        feature = FeatureType.VECTOR_TIMELESS, "test_geom"
        eopatch = EOPatch(bbox=BBox([0, 0, 3, 3], CRS.WGS84))
        eopatch[feature] = gdf

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test.gpkg")
            task = VectorExportTask(feature=feature, path=output_path, driver="GPKG")
            task.execute(eopatch)

            assert os.path.isfile(output_path), "GPKG file was not created"
            result = gpd.read_file(output_path)
            assert len(result) == 3, "Should have 3 features"
            assert list(result.columns) == ["id", "label", "geometry"], "Unexpected columns"
            assert result.crs == gdf.crs, "CRS should be preserved"

    def test_export_geojson(self):
        """Test exporting a vector feature to GeoJSON format."""
        gdf = _create_test_geodataframe()
        feature = FeatureType.VECTOR_TIMELESS, "test_geom"
        eopatch = EOPatch(bbox=BBox([0, 0, 3, 3], CRS.WGS84))
        eopatch[feature] = gdf

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test.geojson")
            task = VectorExportTask(feature=feature, path=output_path, driver="GeoJSON")
            task.execute(eopatch)

            assert os.path.isfile(output_path), "GeoJSON file was not created"
            result = gpd.read_file(output_path)
            assert len(result) == 3, "Should have 3 features"

    def test_export_empty_feature(self):
        """Test that exporting a non-existent feature raises an error."""
        feature = FeatureType.VECTOR_TIMELESS, "nonexistent"
        eopatch = EOPatch(bbox=BBox([0, 0, 3, 3], CRS.WGS84))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test.gpkg")
            task = VectorExportTask(feature=feature, path=output_path)
            with pytest.raises((KeyError, ValueError), match="nonexistent|no data"):
                task.execute(eopatch)

    def test_export_roundtrip_gpkg(self):
        """Test export then import round-trip with GPKG format."""
        gdf = _create_test_geodataframe()
        feature = FeatureType.VECTOR_TIMELESS, "test_geom"
        eopatch = EOPatch(bbox=BBox([0, 0, 3, 3], CRS.WGS84))
        eopatch[feature] = gdf

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "roundtrip.gpkg")
            export_task = VectorExportTask(feature=feature, path=output_path)
            export_task.execute(eopatch)

            # Import back and verify
            import_task = VectorImportTask(feature=feature, path=output_path)
            imported = import_task.execute(bbox=BBox([0, 0, 3, 3], CRS.WGS84))

            assert len(imported[feature]) == 3, "Round-trip should preserve feature count"
            assert imported[feature].crs.to_epsg() == gdf.crs.to_epsg(), "CRS should be preserved in round-trip"

    def test_export_creates_new_eopatch_object(self):
        """Test that export returns the same EOPatch object."""
        gdf = _create_test_geodataframe()
        feature = FeatureType.VECTOR_TIMELESS, "test_geom"
        eopatch = EOPatch(bbox=BBox([0, 0, 3, 3], CRS.WGS84))
        eopatch[feature] = gdf

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test.gpkg")
            task = VectorExportTask(feature=feature, path=output_path)
            result = task.execute(eopatch)

            assert result is eopatch, "Should return the same EOPatch instance"
