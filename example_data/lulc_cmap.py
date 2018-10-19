import matplotlib as mpl
import numpy as np

lulc_cmap = mpl.colors.ListedColormap(['white', 'xkcd:lime', 'xkcd:darkgreen', 'orange', 'xkcd:tan', 
                                       'xkcd:azure', 'xkcd:lightblue', 'xkcd:lavender',
                                       'crimson', 'xkcd:beige', 'black'])
lulc_cmap.set_over('white')
lulc_cmap.set_under('white')

bounds = np.arange(-0.5, 11, 1).tolist()
lulc_norm = mpl.colors.BoundaryNorm(bounds, lulc_cmap.N)
