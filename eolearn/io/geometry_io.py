"""
Module for adding vector data from various sources

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import boto3
import fiona
import geopandas as gpd
from fiona.session import AWSSession
from fs.base import FS
from fs_s3fs import S3FS

from sentinelhub import CRS, BBox, SHConfig

from eolearn.core import EOPatch, EOTask, pickle_fs, unpickle_fs
from eolearn.core.types import Feature
from eolearn.core.utils.fs import get_base_filesystem_and_path, get_full_path

LOGGER = logging.getLogger(__name__)


class VectorImportTask(EOTask):
    """A task for importing (Fiona readable) vector data files into an EOPatch"""

    def __init__(
        self,
        feature: Feature,
        path: str,
        reproject: bool = True,
        clip: bool = False,
        filesystem: FS | None = None,
        config: SHConfig | None = None,
        **kwargs: Any,
    ):
        """
        :param feature: A vector feature into which to import data
        :param path: A path to a dataset containing vector data. It can be either a local path or a path to s3 bucket.
            If `filesystem` parameter is given the path should be relative to the filesystem, otherwise it should
            be an absolute path.
        :param reproject: Should the geometries be transformed to coordinate reference system of the requested bbox?
        :param clip: Should the geometries be clipped to the requested bbox, or should be geometries kept as they are?
        :param filesystem: A filesystem object. If not given it will be created from the path and config credentials.
        :param config: A configuration object with AWS credentials (if not provided, ~/.aws/credentials will be used)
        :param kwargs: Additional args that will be passed to `fiona.open` or `geopandas.read` calls (e.g. layer name)
        """
        if filesystem is None:
            filesystem, path = get_base_filesystem_and_path(path, config=config)
        self.path = path
        self.full_path = get_full_path(filesystem, path)
        self._pickled_filesystem = pickle_fs(filesystem)

        self.fiona_kwargs = kwargs
        self._aws_session = None
        self._dataset_crs: CRS | None = None

        self.feature = self.parse_feature(feature, allowed_feature_types=lambda fty: fty.is_vector())
        self.config = config or SHConfig()
        self.reproject = reproject
        self.clip = clip

    @property
    def aws_session(self) -> AWSSession:
        """Because the session object cannot be pickled this provides the session lazily (i.e. the first time it is
        needed)

        :return: A session for AWS services
        """
        if self._aws_session is None:
            filesystem = unpickle_fs(self._pickled_filesystem)
            if not isinstance(filesystem, S3FS):
                raise ValueError(f"AWS session can only be obtained for S3 filesystem but found {filesystem}")

            boto_session = boto3.session.Session(
                aws_access_key_id=filesystem.aws_access_key_id,
                aws_secret_access_key=filesystem.aws_secret_access_key,
                aws_session_token=filesystem.aws_session_token,
                region_name=filesystem.region,
            )
            self._aws_session = AWSSession(boto_session)

        return self._aws_session

    @property
    def dataset_crs(self) -> CRS:
        """Provides a CRS of dataset, it loads it lazily (i.e. the first time it is needed)"""
        if self._dataset_crs is None:
            is_on_s3 = self.full_path.startswith("s3://")
            env = fiona.Env(session=self.aws_session) if is_on_s3 else nullcontext()
            with env, fiona.open(self.full_path, **self.fiona_kwargs) as features:
                self._dataset_crs = CRS(features.crs)

        return self._dataset_crs

    def _load_vector_data(self, bbox: BBox | None) -> gpd.GeoDataFrame:
        """Loads vector data either from S3 or local path"""
        bbox_bounds = bbox.transform_bounds(self.dataset_crs).geometry.bounds if bbox else None

        if self.full_path.startswith("s3://"):
            with fiona.Env(session=self.aws_session), fiona.open(self.full_path, **self.fiona_kwargs) as features:
                feature_iter = features if bbox_bounds is None else features.filter(bbox=bbox_bounds)

                return gpd.GeoDataFrame.from_features(
                    feature_iter,
                    columns=list(features.schema["properties"]) + ["geometry"],  # noqa: RUF005
                    crs=self.dataset_crs.pyproj_crs(),
                )

        return gpd.read_file(self.full_path, bbox=bbox_bounds, **self.fiona_kwargs)

    def _reproject_and_clip(self, vectors: gpd.GeoDataFrame, bbox: BBox | None) -> gpd.GeoDataFrame:
        """Method to reproject and clip vectors to the EOPatch crs and bbox"""

        if self.reproject:
            if not bbox:
                raise ValueError("To reproject vector data, eopatch.bbox has to be defined!")

            vectors = vectors.to_crs(bbox.crs.pyproj_crs())

        if self.clip:
            if not bbox:
                raise ValueError("To clip vector data, eopatch.bbox has to be defined!")

            bbox_crs = bbox.crs.pyproj_crs()
            if vectors.crs != bbox_crs:
                raise ValueError("To clip, vectors should be in same CRS as EOPatch bbox!")

            extent = gpd.GeoSeries([bbox.geometry], crs=bbox_crs)
            vectors = gpd.clip(vectors, extent, keep_geom_type=True)

        return vectors

    def execute(self, eopatch: EOPatch | None = None, *, bbox: BBox | None = None) -> EOPatch:
        """
        :param eopatch: An existing EOPatch. If none is provided it will create a new one.
        :param bbox: A bounding box for which to load data. By default, if none is provided, it will take a bounding box
            of given EOPatch. If given EOPatch is not provided it will load the entire dataset.
        :return: An EOPatch with an additional vector feature
        """
        if bbox is None and eopatch is not None:
            bbox = eopatch.bbox

        vectors = self._load_vector_data(bbox)
        minx, miny, maxx, maxy = vectors.total_bounds
        final_bbox = bbox or BBox((minx, miny, maxx, maxy), crs=CRS(vectors.crs))

        eopatch = eopatch or EOPatch(bbox=final_bbox)
        if eopatch.bbox is None:
            eopatch.bbox = final_bbox

        eopatch[self.feature] = self._reproject_and_clip(vectors, bbox)

        return eopatch
