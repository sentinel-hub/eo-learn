import pytest
import unittest
import os
import numpy as np

from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector
from eolearn.core import EOPatch


class TestAddSentinelHubCloudMaskTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eop = EOPatch.load(os.path.join(os.path.dirname(__file__), 'eopatch_sample'))

    def test_raises_errors(self):
        classifier = get_s2_pixel_cloud_detector(all_bands=True)
        add_cm = AddCloudMaskTask(classifier, 'bands', cmask_feature='clm')
        self.assertRaises(ValueError, add_cm, self.eop)

    def test_cloud_coverage(self):
        classifier = get_s2_pixel_cloud_detector(all_bands=True)
        # Classifier is run on same resolution as data array
        add_cm = AddCloudMaskTask(classifier, 'ALL_BANDS', cmask_feature='clm', cprobs_feature='clp')
        eop_clm = add_cm(self.eop)
        _, h, w, _ = eop_clm.mask['clm'].shape
        cc = np.sum(eop_clm.mask['clm'][0]) / (w * h)
        ps = np.sum(eop_clm.data['clp'][0]) / (w * h)
        self.assertTrue(eop_clm.mask['clm'].ndim == 4)
        self.assertAlmostEqual(cc, 0.687936507936508, places=4)
        self.assertAlmostEqual(ps, 0.521114510213301, places=4)
        del add_cm, eop_clm
        # Classifier is run on downscaled version of data array
        add_cm = AddCloudMaskTask(classifier, 'ALL_BANDS', cm_size_y=50, cmask_feature='clm', cprobs_feature='clp')
        eop_clm = add_cm(self.eop)
        _, h, w, _ = eop_clm.mask['clm'].shape
        cc = np.sum(eop_clm.mask['clm'][0]) / (w * h)
        ps = np.sum(eop_clm.data['clp'][0]) / (w * h)
        self.assertTrue(eop_clm.mask['clm'].ndim == 4)
        self.assertAlmostEqual(cc, 0.710357142857142, places=4)
        self.assertAlmostEqual(ps, 0.500692345333859, places=4)

    @pytest.mark.xfail
    def test_wms_request(self):
        classifier = get_s2_pixel_cloud_detector(all_bands=False)
        # Classifier is run on new request of data array
        add_cm = AddCloudMaskTask(classifier, 'BANDS-S2-L1C', cm_size_y=50, cmask_feature='clm', cprobs_feature='clp')
        eop_clm = add_cm(self.eop)
        _, h, w, _ = eop_clm.mask['clm'].shape
        cc = np.sum(eop_clm.mask['clm'][0]) / (w * h)
        ps = np.sum(eop_clm.data['clp'][0]) / (w * h)
        self.assertTrue(eop_clm.mask['clm'].ndim == 4)
        self.assertAlmostEqual(cc, 0.737738, places=4)
        self.assertAlmostEqual(ps, 0.520182, places=4)


if __name__ == '__main__':
    unittest.main()
