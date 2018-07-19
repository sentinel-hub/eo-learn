"""
Module for creating plots and visualisations
"""

import matplotlib.pyplot as plt
import numpy as np

from .eodata import FeatureType
from .eotask import EOTask


def bgr_to_rgb(bgr):
    """
    Converts Blue, Green, Red to Red, Green, Blue
    """
    return bgr[..., [2, 1, 0]]


class IndexTracker:
    """ Class to handle slicing of the eopatch """
    # pylint: disable=invalid-name
    def __init__(self, ax, im_seq, single_channel=False, msg=None, colorbar=False):
        """ Class to handle slicing of the eopatch

        :param ax: Axes handle of the plot
        :param im_seq: 2D single channel or RGB temporal sequence to be displayed
        :param single_channel: Flag to indicate whether the images are grayscale or RGB
        :param msg: Message to be displayed as title of the plot
        """
        self.ax = ax
        title = msg if msg is not None else "use scroll wheel to navigate images"
        self.ax.set_title(title)

        self.data = im_seq
        self.slices, _, _ = im_seq.shape[:3]
        self.ind = self.slices//2
        self.single_channel = single_channel

        if self.single_channel:
            self.im = self.ax.imshow(self.data[self.ind, :, :])
        else:
            self.im = self.ax.imshow(self.data[self.ind, :, :, :])
        if colorbar:
            plt.colorbar(self.im)
        self.update()

    def onscroll(self, event):
        """ Action to be taken when an event is triggered

        Event is scroll of the mouse's wheel. This leads to changing the temporal frame displayed.

        :param event: Scroll of mouse wheel
        """
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        """ Update image to be displayed with new time frame """
        if self.single_channel:
            self.im.set_data(self.data[self.ind, :, :])
        else:
            self.im.set_data(self.data[self.ind, :, :, :])
        self.ax.set_ylabel('time frame %s' % self.ind)
        self.im.axes.figure.canvas.draw()


class PatchShowTask(EOTask):
    """ Task to display sequence of 2D images """
    def __init__(self, feature_type=FeatureType.DATA, feature_name='TRUE_COLOR', indices=None):
        """ This task allows to visualise and slice through an eopatch along the temporal dimension.

        The data to display is supposed to have the following dimensions: n_timeframes x n_rows x n_cols (x n_chan).
        If the array is 3D, the sequence is supposed to be single channel.
        If the array is 4D, the last dimension is either equal to 3 (RGB) or the length of indices must be equal to 1
        or 3.

        :param feature_type: Type of feature to display from eopatch. Default is `Featuretype.DATA`
        :type feature_type: FeatureType
        :param feature_name: Name of feature to display. Default is `TRUE_COLOR`
        :type feature_name: str
        :param indices: Indices of channels to be displayed in multi-channel EOPatches. Length of indices must be either
                        1 or 3. Default is `None`
        :type indices: list
        """
        self.feature_type = feature_type
        self.feature_name = feature_name
        self.indices = indices

    def _get_data_to_display(self, eopatch):
        """ Perform checks on dimensionality of data to make it suitable for display

        :param eopatch: Input eopatch
        :return: Array to display and whether it is single channel or not
        """
        image_seq = eopatch[self.feature_type.value][self.feature_name]
        single_channel = False
        if image_seq.ndim == 3:
            # single channel
            single_channel = True
        elif image_seq.ndim == 4:
            # If multi-channel make sure indexing is correct
            n_channels = image_seq.shape[-1]
            if n_channels == 1:
                image_seq = np.squeeze(image_seq, axis=-1)
                single_channel = True
            elif n_channels == 3:
                pass
            else:
                if (self.indices is not None) and ((len(self.indices) == 1) or (len(self.indices) == 3)):
                    image_seq = image_seq[..., self.indices]
                    if len(self.indices) == 1:
                        image_seq = np.squeeze(image_seq, axis=-1)
                        single_channel = True
                else:
                    raise ValueError("Specify valid indices for multi-channel EOPatch")
        else:
            raise ValueError("Unsupported format for EOPatch")
        return image_seq, single_channel

    def execute(self, eopatch, title=None, colorbar=False):
        """ Show data and scroll through. Currently not working from Jupyter notebooks

        :param eopatch: Image sequence to display. Can be single channel, RGB or multi-channel. If multi-channel,
                        indices must be specified.
        :type eopatch: numpy array
        :param title: String to be displayed as title of the plot
        :type title: string
        :param colorbar: Whether to add colorbar to plot or not. Default is ``False``
        :type colorbar: bool
        :return: Same input eopatch
        """
        image_seq, single_channel = self._get_data_to_display(eopatch)
        # clip to positive values
        if image_seq.dtype is np.float:
            image_seq = np.clip(image_seq, 0, 1)
        elif image_seq.dtype is np.int:
            image_seq = np.clip(image_seq, 0, 255)
        # Call IndexTracker and visualise time frames
        fig, axis = plt.subplots(1, 1)
        tracker = IndexTracker(axis, image_seq,
                               single_channel=single_channel,
                               msg=title,
                               colorbar=colorbar)
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()
        return eopatch
