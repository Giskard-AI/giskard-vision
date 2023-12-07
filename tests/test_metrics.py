import numpy as np

from loreal_poc.tests.performance import _calculate_es


TEST_ARRAY_A_2D = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
TEST_ARRAY_B_2D = [[4.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 3.0]]
TEST_ARRAY_A_3D = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]
TEST_ARRAY_B_3D = [[4.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 0.0], [2.0, 3.0, 1.0]]


def test_calculate_es_2d():
    a = np.asarray(TEST_ARRAY_A_2D)
    b = np.asarray(TEST_ARRAY_B_2D)
    c = np.asarray([np.sqrt((p_a[0] - p_b[0]) ** 2 + (p_a[1] - p_b[1]) ** 2) for p_a, p_b in zip(a, b)])
    calculated = _calculate_es(np.asarray([a]), np.asarray([b]))
    assert np.all(np.isclose(np.asarray([c]), calculated))


def test_calculate_es_3d():
    a = np.asarray(TEST_ARRAY_A_3D)
    b = np.asarray(TEST_ARRAY_B_3D)
    c = np.asarray(
        [np.sqrt((p_a[0] - p_b[0]) ** 2 + (p_a[1] - p_b[1]) ** 2 + (p_a[2] - p_b[2]) ** 2) for p_a, p_b in zip(a, b)]
    )
    calculated = _calculate_es(np.asarray([a]), np.asarray([b]))
    assert np.all(np.isclose(np.asarray([c]), calculated))
