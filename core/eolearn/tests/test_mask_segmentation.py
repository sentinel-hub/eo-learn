import unittest


class TestlabelMaskProcessor(unittest.TestCase):
    pass


if __name__ == '__main__':
    pass


"""
label_mask = MaskSegmentation('azerbaijan.tif')

crs = label_mask.crs
labels = label_mask.labels # the set of all labels

# iterate over connected components
for cc in label_mask:
    # 100 randomly chosen points from cc
    pts = cc.sample(100)
    # Now I want to be able to request actual data for each point (via Sentinel Hub)


number_of_ccs = len(label_mask.components)

grass_cc_idxs = label_mask.label2cc(GRASS) # returns set of indices of CCs labelled GRASS
"""

"""
Randomly sample n points from polygon P.

Use the rectangular decomposition RD(P) = R1, R2, ..., Rk with weights w1, w2, ..., wk; here wi = Area(Ri) / Area(P)

Weighted sampling of i1, i2, ..., in with respective weights
Convert to (j1, m1), ..., (jp, mp), i.e., rectangle with index j1 was drawn m1 times

for each i=1 to p do
    sample mi points from Ri
"""
