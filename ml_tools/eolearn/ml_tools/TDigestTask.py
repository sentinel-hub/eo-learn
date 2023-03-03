"""
The module provides an EOTask for the computation of a T-Digest representation of an EOPatch.

Credits:
Copyright (c) 2023 Michael Engel

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""
from itertools import product
import numpy as np

from eolearn.core import EOTask
import tdigest as td


class TDigestTask(EOTask):
    """
    An EOTask to compute the T-Digest representation of a chosen feature of an EOPatch.
    It integrates the [T-Digest algorithm by Ted Dunning](https://arxiv.org/abs/1902.04023) to efficiently compute quantiles of the underlying dataset into eo-learn.
    The output features of the tasks may be merged to compute a representation of the complete dataset.
    That enables quantile based normalisation or statistical analysis of datasets larger than RAM in EO.
    """
    def __init__(self, in_feature, out_feature, *args, mode=None, pixelwise=True, **kwargs):
        """
        :param in_feature: The input feature to compute the T-Digest representation for.
        :type in_feature: (FeatureType, str)
        :param out_feature: The output feature where to save the T-Digest representation of the chosen feature.
        :type out_feature: (FeatureType, str)
        :param mode: The mode to apply to the timestamps and bands.
        - The 'standard' mode computes the T-Digest representation for each band accumulating timestamps.
        - The 'timewise' mode computes the T-Digest representation for each band and timestamp of the chosen feature.
        - The 'monthly' mode computes the T-Digest representation for each band accumulating the timestamps per month.
        - The 'weekly' mode computes the T-Digest representation for each band accumulating the timestamps per seven days.
        - The 'daily' mode computes the T-Digest representation for each band accumulating the timestamps per day.
        - The 'total' mode computes the total T-Digest representation of the whole feature accumulating all timestamps, bands and pixels.
        :type mode: str
        :param pixelwise: Decider whether to compute the T-Digest representation accumulating pixels or per pixel.
        :type pixelwise: boolean
        :param args: arguments for init of EOTask base class.
        :type args: arguments
        :param kwargs: keyword arguments for init of EOTask base class.
        :type kwargs: keyword arguments
        """
        super().__init__(*args, **kwargs)

        # check input and output features
        if isinstance(in_feature,list):
            self.in_feature = in_feature
            if isinstance(out_feature,list):
                assert len(in_feature)==len(out_feature)
                self.out_feature = out_feature
            else:
                raise RuntimeError("TDigestTask: you cannot give a list of input features without specifying the a list of output features of the same length!")
        else:
            self.in_feature = [in_feature]
            if isinstance(out_feature,list):
                raise RuntimeError("TDigestTask: you cannot get a list of T-Digest representations from a single feature!")
            else:
                self.out_feature = [out_feature]

        # check pixelwise parameter
        if isinstance(pixelwise,list):
            if len(pixelwise)==len(self.out_feature):
                self.pixelwise = pixelwise
            else:
                assert len(pixelwise)==len(self.out_feature)
        else:
            self.pixelwise = [pixelwise]*len(self.out_feature)

        # set mode
        self.mode = mode

    def execute(self, eopatch):
        """
        Execute method that computes the TDigest of the chosen features.
  
        :param eopatch: EOPatch which the chosen input feature already exists
        :type eopatch: EOPatch
        """

        # standard mode
        if self.mode is None or self.mode=='standard':
            for in_feature,out_feature,pixelwise in zip(self.in_feature, self.out_feature, self.pixelwise):
                shape = np.array(eopatch[in_feature].shape)
                if pixelwise:
                    eopatch[out_feature] = np.empty(shape[1:],dtype=object)
                    if len(shape)==4: # add feature is temporal check
                        for i,j,k in product(range(shape[1]),range(shape[2]),range(shape[3])):
                            eopatch[out_feature][i,j,k] = td.TDigest()
                            eopatch[out_feature][i,j,k].batch_update(eopatch[in_feature][:,i,j,k].flatten())
                    elif len(shape)==3: # add feature is timeless check
                        for i,j,k in product(range(shape[0]),range(shape[1]),range(shape[2])):
                            eopatch[out_feature][i,j,k] = td.TDigest()
                            eopatch[out_feature][i,j,k].batch_update(eopatch[in_feature][i,j,k].flatten())
                    else:
                        raise RuntimeError(f"TDigestTask: {self.mode} mode only defined for DATA, DATA_TIMELESS, MASK or MASK_TIMELESS feature types!")

                else:
                    eopatch[out_feature] = np.empty(shape[-1],dtype=object)
                    for k in range(shape[-1]):
                        eopatch[out_feature][k] = td.TDigest()
                        eopatch[out_feature][k].batch_update(eopatch[in_feature][...,k].flatten())

        # timewise mode
        elif self.mode=='timewise':
            for time_,timestamp in enumerate(eopatch["timestamp"]):
                for in_feature,out_feature,pixelwise in zip(self.in_feature, self.out_feature, self.pixelwise):
                    out_feature_type, out_feature_name = out_feature
                    out_feature = (out_feature_type, out_feature_name+f"_{timestamp.strftime('%Y-%m-%d-%H-%M-%S')}")

                    shape = np.array(eopatch[in_feature].shape)
                    if pixelwise:
                        eopatch[out_feature] = np.empty(shape,dtype=object)
                        if len(shape)==4: # add feature is temporal check
                            for i,j,k in product(range(shape[1]),range(shape[2]),range(shape[3])):
                                eopatch[out_feature][time_,i,j,k] = td.TDigest()
                                eopatch[out_feature][time_,i,j,k].batch_update(eopatch[in_feature][time_,i,j,k].flatten())
                        else:
                            raise RuntimeError(f"TDigestTask: {self.mode} mode only defined for DATA or MASK feature types with time component!")

                    else:
                        eopatch[out_feature] = np.empty(shape[[0,-1]],dtype=object)
                        for k in range(shape[-1]):
                            eopatch[out_feature][time_,k] = td.TDigest()
                            eopatch[out_feature][time_,k].batch_update(eopatch[in_feature][time_,...,k].flatten())

        # monthly mode
        elif self.mode=="monthly":
            midx = []
            for month_ in range(12):
                midx.append(np.array([timestamp.month==month_+1 for timestamp in eopatch['timestamp']]))

            for in_feature,out_feature,pixelwise in zip(self.in_feature, self.out_feature, self.pixelwise):
                shape = np.array(eopatch[in_feature].shape)
                if pixelwise:
                    eopatch[out_feature] = np.empty([12,*shape[1:]],dtype=object)
                    if len(shape)==4: # add feature is temporal check
                        for month_,i,j,k in product(range(12),range(shape[1]),range(shape[2]),range(shape[3])):
                            eopatch[out_feature][month_,i,j,k] = td.TDigest()
                            eopatch[out_feature][month_,i,j,k].batch_update(eopatch[in_feature][midx[month_],i,j,k].flatten())
                    else:
                        raise RuntimeError(f"TDigestTask: {self.mode} mode only defined for DATA or MASK feature types with time component!")

                else:
                    eopatch[out_feature] = np.empty([12,shape[-1]],dtype=object)
                    for month_,k in product(range(12),range(shape[-1])):
                        eopatch[out_feature][month_,k] = td.TDigest()
                        eopatch[out_feature][month_,k].batch_update(eopatch[in_feature][midx[month_],...,k].flatten())

        # weekly mode
        elif self.mode=="weekly":
            raise NotImplementedError(f"TDigestTask: {self.mode} mode not implemented yet!")

        # daily mode
        elif self.mode=="daily":
            raise NotImplementedError(f"TDigestTask: {self.mode} mode not implemented yet!")

        # total mode
        elif self.mode=="total":
            for in_feature,out_feature,pixelwise in zip(self.in_feature, self.out_feature, self.pixelwise):
                if pixelwise:
                    raise NotImplementedError(f"TDigestTask: pixelwise for {self.mode} mode not implemented yet!")

                else:
                    eopatch[out_feature] = td.TDigest()
                    eopatch[out_feature].batch_update(eopatch[in_feature].flatten())

        # errorneous modes
        else:
            raise RuntimeError(f"TDigestTask: mode {self.mode} not implemented!")

        # return
        return eopatch
