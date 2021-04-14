"""
Module for adding vector data from various sources

Credits:
Copyright (c) 2017-2021 Matej Aleksandrov, Matej Batič (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import logging

import boto3
import fiona
import geopandas as gpd
from eolearn.core import EOPatch, EOTask, FeatureType
from fiona.session import AWSSession

from sentinelhub import BBox, CRS, GeopediaFeatureIterator, SHConfig

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')


class VectorImportTask(EOTask):
    """ A task for importing (Fiona readable) vector data files into an EOPatch
    """
    def __init__(self, feature, path, reproject=True, clip=False, config=None, **kwargs):
        """
        :param feature: A vector feature into which to import data
        :type feature: (FeatureType, str)
        :param path: A path to a dataset containing vector data. It can be either a local path or a path to s3 bucket
        :type path: str
        :param reproject: Should the geometries be transformed to EOPatch coordinate reference system?
        :type reproject: bool, default = True
        :param clip: Should the geometries be clipped to EOPatch bbox, or should be geometries kept as they are?
        :type clip: bool, default = False
        :param config: A configuration object containing AWS credentials
        :type config: SHConfig
        :param kwargs: Additional args that will be passed to `fiona.open` call (e.g. layer='GPKG_LAYER')
        """
        self.feature = next(self._parse_features(
            feature, allowed_feature_types=[FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS]
        )())
        self.path = path
        self.config = config or SHConfig()
        self.reproject = reproject
        self.clip = clip

        self._aws_session = None
        self._dataset_crs = None

        self.additional_args = kwargs

    @property
    def aws_session(self):
        """ Because the session object cannot be pickled this provides the session lazily (i.e. the first time it is
        needed)

        :return: A session for AWS services
        :rtype: AWSSession
        """
        if self._aws_session is None:
            session_credentials = {}
            if self.config:
                session_credentials['aws_access_key_id'] = self.config.aws_access_key_id
                session_credentials['aws_secret_access_key'] = self.config.aws_secret_access_key

            boto_session = boto3.session.Session(**session_credentials)
            self._aws_session = AWSSession(boto_session)

        return self._aws_session

    @property
    def dataset_crs(self):
        """ Provides a CRS of dataset, it loads it lazily (i.e. the first time it is needed)

        :return: Dataset's CRS
        :rtype: CRS
        """
        if self._dataset_crs is None:
            if self.path.startswith('s3://'):
                with fiona.Env(session=self.aws_session):
                    self._read_crs()
            else:
                self._read_crs()

        return self._dataset_crs

    def _read_crs(self):
        """ Reads information about CRS from a dataset
        """
        with fiona.open(self.path, encoding='utf-8') as features:
            self._dataset_crs = CRS(features.crs)

    def _prepare_bbox(self, bbox):
        """ This makes sure that the bounding box is in correct CRS. The final bounds can be larger than given bbox
        """
        if bbox is None:
            return None

        return bbox.transform_bounds(self.dataset_crs)

    def _load_vector_data(self, bbox):
        """ Loads vector data either from S3 or local path
        """
        bbox_bounds = bbox.geometry.bounds if bbox else None
        if self.path.startswith('s3://'):
            with fiona.Env(session=self.aws_session):
                with fiona.open(self.path, encoding='utf-8', **self.additional_args) as features:
                    feature_iter = features if bbox_bounds is None else features.filter(bbox=bbox_bounds)

                    return gpd.GeoDataFrame.from_features(
                        feature_iter,
                        columns=list(features.schema['properties']) + ['geometry'],
                        crs=features.crs
                    )

        return gpd.read_file(self.path, bbox=bbox_bounds, **self.additional_args)

    def _reproject_and_clip(self, vectors, bbox):
        """Method to reproject and clip vectors to the EOPatch crs and bbox"""
        if not bbox and (self.reproject or self.clip):
            raise ValueError('To clip or reproject vector data, eopatch.bbox has to be defined!')

        if self.reproject:
            vectors = vectors.to_crs(bbox.crs.pyproj_crs())

        if self.clip:
            bbox_crs = bbox.crs.pyproj_crs()
            if vectors.crs != bbox_crs:
                raise ValueError(f'To clip, vectors should be in same CRS as EOPatch bbox!')

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

        prepared_bbox = self._prepare_bbox(bbox)
        vectors = self._load_vector_data(prepared_bbox)

        eopatch[self.feature] = self._reproject_and_clip(vectors, eopatch.bbox)

        if not eopatch.bbox:
            eopatch.bbox = BBox(list(eopatch[self.feature].total_bounds), crs=eopatch[self.feature].crs.to_epsg())

        return eopatch


class GeopediaVectorImportTask(VectorImportTask):
    """ A task for importing Geopedia features into EOPatch vector features
    """
    def __init__(self, feature, geopedia_table, reproject=True, clip=False, config=None, **kwargs):
        self.geopedia_table = geopedia_table
        super().__init__(feature=feature, path=None, reproject=reproject, clip=clip, config=config, **kwargs)

    def _prepare_bbox(self, bbox):
        """ This makes sure that the bounding box is in correct CRS. The final bounds can be larger than given bbox
        """
        if bbox is None:
            self._dataset_crs = CRS.WGS84  # geopedia always returns data in WGS84
            return None

        self._dataset_crs = bbox.crs
        return bbox.transform_bounds(CRS.POP_WEB)

    def _load_vector_data(self, bbox):
        """ Loads vector data from geopedia table
        """

        geopedia_iterator = GeopediaFeatureIterator(
            layer=self.geopedia_table,
            bbox=bbox,
            offset=0,
            gpd_session=None,
            config=self.config,
            **self.additional_args
        )
        geopedia_features = list(geopedia_iterator)

        geometry = geopedia_features[0].get('geometry')
        if not geometry:
            raise ValueError(f'Geopedia table "{self.geopedia_table}" does not contain geometries!')

        crs = CRS(geometry['crs']['properties']['name'])
        vectors = gpd.GeoDataFrame.from_features(geopedia_features, crs=crs.pyproj_crs())

        return vectors.to_crs(self.dataset_crs.pyproj_crs())


class GeoDBVectorImportTask(VectorImportTask):
    """ A task for importing vector data from geoDB into EOPatch
    """
    def __init__(self, feature, geodb_client, geodb_collection, geodb_db,
                 reproject=True, clip=False, config=None,
                 **kwargs):
        """
        :param feature: A vector feature into which to import data
        :param geodb_client: an instance of xcube_geodb.core.geodb.GeoDBClient
        :param geodb_collection: The name of the collection to be quried
        :param geodb_db: The name of the database the collection resides in [current database]
        :param reproject: Should the geometries be transformed to EOPatch coordinate reference system? (default = True)
        :param clip: Should the geometries be clipped to EOPatch bbox, or should be geometries kept as they are?
            (default = False)
        :param config: A configuration object containing AWS credentials (not needed for geoDB, default = None)
        :param kwargs: Additional args that will be passed to `geodb_client.get_collection_by_bbox` call
            (e.g. where="id>-1", operator="and")
        """
        self.geodb_client = geodb_client
        self.geodb_db = geodb_db
        self.geodb_collection = geodb_collection
        self.additional_args = kwargs

        super().__init__(feature=feature, path=None, reproject=reproject, clip=clip, config=config)

    @property
    def dataset_crs(self):
        """ Provides a "crs" of dataset, loads it lazily (i.e. the first time it is needed)

        :return: Dataset's CRS
        :rtype: CRS
        """
        if self._dataset_crs is None:
            # this might fail; geodb supports larger number of coordinate reference systems than CRS enum
            srid = self.geodb_client.get_collection_srid(
                collection=self.geodb_collection,
                database=self.geodb_db
            )
            self._dataset_crs = CRS(f'epsg:{srid}')

        return self._dataset_crs

    def _load_vector_data(self, bbox):
        """ Loads vector data from geoDB table
        """

        return self.geodb_client.get_collection_by_bbox(
            collection=self.geodb_collection,
            database=self.geodb_db,
            bbox=bbox.geometry.bounds,
            comparison_mode="contains",
            bbox_crs=self.dataset_crs.epsg,
            **self.additional_args
        )


