"""
Module for adding vector data from various sources

Credits:
Copyright (c) 2017-2022 Matej Aleksandrov, Matej Batič, Žiga Lukšič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import abc
import logging

import boto3
import fiona
import geopandas as gpd
from fiona.session import AWSSession

from sentinelhub import CRS, GeopediaFeatureIterator, SHConfig

from eolearn.core import EOPatch, EOTask, FeatureTypeSet

LOGGER = logging.getLogger(__name__)


class _BaseVectorImportTask(EOTask, metaclass=abc.ABCMeta):
    """Base Vector Import Task, implementing common methods"""

    def __init__(self, feature, reproject=True, clip=False, config=None):
        """
        :param feature: A vector feature into which to import data
        :type feature: (FeatureType, str)
        :param reproject: Should the geometries be transformed to coordinate reference system of the requested bbox?
        :type reproject: bool, default = True
        :param clip: Should the geometries be clipped to the requested bbox, or should be geometries kept as they are?
        :type clip: bool, default = False
        :param config: A configuration object with credentials
        :type config: SHConfig
        """
        self.feature = self.parse_feature(feature, allowed_feature_types=FeatureTypeSet.VECTOR_TYPES)
        self.config = config or SHConfig()
        self.reproject = reproject
        self.clip = clip

    @abc.abstractmethod
    def _load_vector_data(self, bbox):
        """Loads vector data given a bounding box"""

    def _reproject_and_clip(self, vectors, bbox):
        """Method to reproject and clip vectors to the EOPatch crs and bbox"""
        if not bbox and (self.reproject or self.clip):
            raise ValueError("To clip or reproject vector data, eopatch.bbox has to be defined!")

        if self.reproject:
            vectors = vectors.to_crs(bbox.crs.pyproj_crs())

        if self.clip:
            bbox_crs = bbox.crs.pyproj_crs()
            if vectors.crs != bbox_crs:
                raise ValueError("To clip, vectors should be in same CRS as EOPatch bbox!")

            extent = gpd.GeoSeries([bbox.geometry], crs=bbox_crs)
            vectors = gpd.clip(vectors, extent, keep_geom_type=True)

        return vectors

    def execute(self, eopatch=None, *, bbox=None):
        """
        :param eopatch: An existing EOPatch. If none is provided it will create a new one.
        :type eopatch: EOPatch
        :param bbox: A bounding box for which to load data. By default, if none is provided, it will take a bounding box
            of given EOPatch. If given EOPatch is not provided it will load the entire dataset.
        :return: An EOPatch with an additional vector feature
        :rtype: EOPatch
        """
        eopatch = eopatch or EOPatch()
        bbox = bbox or eopatch.bbox

        if not eopatch.bbox:
            eopatch.bbox = bbox

        vectors = self._load_vector_data(bbox)
        eopatch[self.feature] = self._reproject_and_clip(vectors, bbox)

        return eopatch


class VectorImportTask(_BaseVectorImportTask):
    """A task for importing (Fiona readable) vector data files into an EOPatch"""

    def __init__(self, feature, path, reproject=True, clip=False, config=None, **kwargs):
        """
        :param feature: A vector feature into which to import data
        :type feature: (FeatureType, str)
        :param path: A path to a dataset containing vector data. It can be either a local path or a path to s3 bucket
        :type path: str
        :param reproject: Should the geometries be transformed to coordinate reference system of the requested bbox?
        :type reproject: bool, default = True
        :param clip: Should the geometries be clipped to the requested bbox, or should be geometries kept as they are?
        :type clip: bool, default = False
        :param config: A configuration object with AWS credentials (if not provided, ~/.aws/credentials will be used)
        :type config: SHConfig
        :param kwargs: Additional args that will be passed to `fiona.open` or `geopandas.read` calls (e.g. layer name)
        """
        self.path = path
        self.fiona_kwargs = kwargs
        self._aws_session = None
        self._dataset_crs = None

        super().__init__(feature=feature, reproject=reproject, clip=clip, config=config)

    @property
    def aws_session(self):
        """Because the session object cannot be pickled this provides the session lazily (i.e. the first time it is
        needed)

        :return: A session for AWS services
        :rtype: AWSSession
        """
        if self._aws_session is None:
            boto_session = boto3.session.Session(
                aws_access_key_id=self.config.aws_access_key_id or None,
                aws_secret_access_key=self.config.aws_secret_access_key or None,
                aws_session_token=self.config.aws_session_token or None,
            )
            self._aws_session = AWSSession(boto_session)

        return self._aws_session

    @property
    def dataset_crs(self):
        """Provides a CRS of dataset, it loads it lazily (i.e. the first time it is needed)

        :return: Dataset's CRS
        :rtype: CRS
        """
        if self._dataset_crs is None:
            if self.path.startswith("s3://"):
                with fiona.Env(session=self.aws_session):
                    self._read_crs()
            else:
                self._read_crs()

        return self._dataset_crs

    def _read_crs(self):
        """Reads information about CRS from a dataset"""
        with fiona.open(self.path, **self.fiona_kwargs) as features:
            self._dataset_crs = CRS(features.crs)

    def _load_vector_data(self, bbox):
        """Loads vector data either from S3 or local path"""
        bbox_bounds = bbox.transform_bounds(self.dataset_crs).geometry.bounds if bbox else None

        if self.path.startswith("s3://"):
            with fiona.Env(session=self.aws_session):
                with fiona.open(self.path, **self.fiona_kwargs) as features:
                    feature_iter = features if bbox_bounds is None else features.filter(bbox=bbox_bounds)

                    return gpd.GeoDataFrame.from_features(
                        feature_iter,
                        columns=list(features.schema["properties"]) + ["geometry"],
                        crs=self.dataset_crs.pyproj_crs(),
                    )

        return gpd.read_file(self.path, bbox=bbox_bounds, **self.fiona_kwargs)


class GeopediaVectorImportTask(_BaseVectorImportTask):
    """A task for importing `Geopedia <https://geopedia.world>`__ features into EOPatch vector features"""

    def __init__(self, feature, geopedia_table, reproject=True, clip=False, **kwargs):
        """
        :param feature: A vector feature into which to import data
        :type feature: (FeatureType, str)
        :param geopedia_table: A Geopedia table from which to retrieve features
        :type geopedia_table: str or int
        :param reproject: Should the geometries be transformed to coordinate reference system of the requested bbox?
        :type reproject: bool, default = True
        :param clip: Should the geometries be clipped to the requested bbox, or should be geometries kept as they are?
        :type clip: bool, default = False
        :param kwargs: Additional args that will be passed to `GeopediaFeatureIterator`
        """
        self.geopedia_table = geopedia_table
        self.geopedia_kwargs = kwargs
        self.dataset_crs = None
        super().__init__(feature=feature, reproject=reproject, clip=clip)

    def _load_vector_data(self, bbox):
        """Loads vector data from geopedia table"""
        prepared_bbox = bbox.transform_bounds(CRS.POP_WEB) if bbox else None

        geopedia_iterator = GeopediaFeatureIterator(
            layer=self.geopedia_table,
            bbox=prepared_bbox,
            offset=0,
            gpd_session=None,
            config=self.config,
            **self.geopedia_kwargs,
        )
        geopedia_features = list(geopedia_iterator)

        geometry = geopedia_features[0].get("geometry")
        if not geometry:
            raise ValueError(f'Geopedia table "{self.geopedia_table}" does not contain geometries!')

        self.dataset_crs = CRS(geometry["crs"]["properties"]["name"])  # always WGS84
        return gpd.GeoDataFrame.from_features(geopedia_features, crs=self.dataset_crs.pyproj_crs())
