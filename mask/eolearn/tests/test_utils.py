import numpy as np
import pytest

from eolearn.mask.utils import ResizeLib, ResizeMethod, map_over_axis, spatially_resize_image


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


def test_map_over_axis():
    data = np.ones((5, 10, 10))
    result = map_over_axis(data, lambda x: np.zeros((7, 20)), axis=0)
    assert result.shape == (5, 7, 20)
    result = map_over_axis(data, lambda x: np.zeros((7, 20)), axis=1)
    assert result.shape == (7, 10, 20)
    result = map_over_axis(data, lambda x: np.zeros((5, 10)), axis=1)
    assert result.shape == (5, 10, 10)


@pytest.mark.parametrize("method", ResizeMethod)
@pytest.mark.parametrize("library", ResizeLib)
@pytest.mark.parametrize("dtype", (np.float32, np.int32, bool))
@pytest.mark.parametrize("new_size", [(50, 50), (35, 39), (271, 271)])
def test_spatially_resize_image_new_size(method, library, dtype, new_size):
    """Test that all methods and backends are able to downscale and upscale images of various dtypes."""
    if library is ResizeLib.CV2:
        if dtype == np.int32 and method is ResizeMethod.CUBIC or dtype == bool:
            return

    old_shape = (111, 111)
    data_2d = np.arange(np.prod(old_shape)).astype(dtype).reshape(old_shape)
    result = spatially_resize_image(data_2d, new_size, resize_method=method, resize_library=library)
    assert result.shape == new_size

    old_shape = (111, 111, 3)
    data_3d = np.arange(np.prod(old_shape)).astype(dtype).reshape(old_shape)
    result = spatially_resize_image(data_3d, new_size, resize_method=method, resize_library=library)
    assert result.shape == (*new_size, 3)

    old_shape = (5, 111, 111, 3)
    data_4d = np.arange(np.prod(old_shape)).astype(dtype).reshape(old_shape)
    result = spatially_resize_image(data_4d, new_size, resize_method=method, resize_library=library)
    assert result.shape == (5, *new_size, 3)

    old_shape = (2, 1, 111, 111, 3)
    data_5d = np.arange(np.prod(old_shape)).astype(dtype).reshape(old_shape)
    result = spatially_resize_image(
        data_5d, new_size, resize_method=method, spatial_axes=(2, 3), resize_library=library
    )
    assert result.shape == (2, 1, *new_size, 3)


@pytest.mark.parametrize("method", ResizeMethod)
@pytest.mark.parametrize("library", ResizeLib)
@pytest.mark.parametrize("new_size", [(50, 50), (217, 271)])
def test_spatially_resize_image(method, library, new_size, test_image):
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
