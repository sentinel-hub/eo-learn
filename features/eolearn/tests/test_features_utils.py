from typing import Tuple, Union

import numpy as np
import pytest

from eolearn.features.utils import ResizeLib, ResizeMethod, spatially_resize_image


@pytest.mark.parametrize("method", ResizeMethod)
@pytest.mark.parametrize("library", ResizeLib)
@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.uint8, bool])
@pytest.mark.parametrize("new_size", [(50, 50), (35, 39), (271, 271)])
def test_spatially_resize_image_new_size(
    method: ResizeMethod, library: ResizeLib, dtype: Union[np.dtype, type], new_size: Tuple[int, int]
):
    """Test that all methods and backends are able to downscale and upscale images of various dtypes."""
    if library is ResizeLib.CV2:  # noqa: SIM102
        if np.issubdtype(dtype, np.integer) and method is ResizeMethod.CUBIC or dtype == bool:
            return

    old_shape = (111, 111)
    data_2d = np.arange(np.prod(old_shape)).astype(dtype).reshape(old_shape)
    result = spatially_resize_image(data_2d, new_size, resize_method=method, resize_library=library)
    assert result.shape == new_size
    assert result.dtype == dtype

    old_shape = (111, 111, 3)
    data_3d = np.arange(np.prod(old_shape)).astype(dtype).reshape(old_shape)
    result = spatially_resize_image(data_3d, new_size, resize_method=method, resize_library=library)
    assert result.shape == (*new_size, 3)
    assert result.dtype == dtype

    old_shape = (5, 111, 111, 3)
    data_4d = np.arange(np.prod(old_shape)).astype(dtype).reshape(old_shape)
    result = spatially_resize_image(data_4d, new_size, resize_method=method, resize_library=library)
    assert result.shape == (5, *new_size, 3)
    assert result.dtype == dtype

    old_shape = (2, 1, 111, 111, 3)
    data_5d = np.arange(np.prod(old_shape)).astype(dtype).reshape(old_shape)
    result = spatially_resize_image(
        data_5d, new_size, resize_method=method, spatial_axes=(2, 3), resize_library=library
    )
    assert result.shape == (2, 1, *new_size, 3)
    assert result.dtype == dtype


@pytest.mark.parametrize("method", ResizeMethod)
@pytest.mark.parametrize("library", ResizeLib)
@pytest.mark.parametrize("scale_factors", [(2, 2), (0.25, 0.25)])
def test_spatially_resize_image_scale_factors(
    method: ResizeMethod, library: ResizeLib, scale_factors: Tuple[float, float]
):
    height, width = 120, 120
    old_shape = (height, width, 3)
    data_3d = np.arange(np.prod(old_shape)).astype(np.float32).reshape(old_shape)

    result = spatially_resize_image(data_3d, scale_factors=scale_factors, resize_method=method, resize_library=library)

    assert result.shape == (height * scale_factors[0], width * scale_factors[1], 3)


@pytest.mark.parametrize("library", [ResizeLib.PIL])
@pytest.mark.parametrize(
    "dtype",
    [
        bool,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        int,
        np.float16,
        np.float32,
        np.float64,
        float,
    ],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_spatially_resize_image_dtype(library: ResizeLib, dtype: Union[np.dtype, type]):
    # Warnings occur due to lossy casting in the downsampling procedure
    old_shape = (111, 111)
    data_2d = np.arange(np.prod(old_shape)).astype(dtype).reshape(old_shape)
    result = spatially_resize_image(data_2d, (50, 50), resize_library=library)
    assert result.dtype == dtype


@pytest.fixture(name="test_image", scope="module")
def test_image_fixture():
    """Image with a distinct value in each of the quadrants. In the bottom right quadrant there is a special layer."""
    example = np.zeros((3, 100, 100, 4))
    example[:, 50:, :50, :] = 1
    example[:, :50, 50:, :] = 2
    example[:, 50:, 50:, :] = 3
    example[0, 50:, 50:, 0] = 10
    example[0, 50:, 50:, 1] = 20
    example[0, 50:, 50:, 2] = 30
    example[0, 50:, 50:, 3] = 40
    return example.astype(np.float32)


@pytest.mark.parametrize("method", ResizeMethod)
@pytest.mark.parametrize("library", ResizeLib)
@pytest.mark.parametrize("new_size", [(50, 50), (217, 271)])
def test_spatially_resize_image_correctness(
    method: ResizeMethod, library: ResizeLib, new_size: Tuple[int, int], test_image: np.ndarray
):
    """Test that resizing is correct on a very basic example. It tests midpoints of the 4 quadrants."""
    height, width = new_size
    x1, x2 = width // 4, 3 * width // 4
    y1, y2 = height // 4, 3 * height // 4

    result = spatially_resize_image(test_image, new_size, resize_method=method, resize_library=library)
    assert result.shape == (3, *new_size, 4)
    assert result[1, x1, y1, :] == pytest.approx([0, 0, 0, 0])
    assert result[2, x2, y1, :] == pytest.approx([1, 1, 1, 1])
    assert result[1, x1, y2, :] == pytest.approx([2, 2, 2, 2])
    assert result[0, x2, y2, :] == pytest.approx([10, 20, 30, 40]), "The first temporal layer of image is incorrect."
    assert result[1, x2, y2, :] == pytest.approx([3, 3, 3, 3])
