import numpy as np
from loreal_poc.tests.performance import (
    _calculate_es,
    _calculate_d_outers,
    LEFT_EYE_LEFT_LANDMARK,
    RIGHT_EYE_RIGHT_LANDMARK,
)


TEST_ARRAY_A_2D = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
TEST_ARRAY_B_2D = [[4.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 3.0]]
TEST_ARRAY_A_3D = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]
TEST_ARRAY_B_3D = [[4.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 0.0], [2.0, 3.0, 1.0]]

TEST_MARKS = [
    [446.000, 91.000],
    [449.459, 119.344],
    [450.957, 150.614],
    [460.552, 176.986],
    [471.486, 202.157],
    [488.087, 226.842],
    [506.016, 246.438],
    [524.662, 263.865],
    [553.315, 271.435],
    [578.732, 266.260],
    [599.361, 248.966],
    [615.947, 220.651],
    [627.439, 197.999],
    [635.375, 179.064],
    [642.063, 156.371],
    [647.302, 124.753],
    [646.518, 92.944],
    [470.271, 117.870],
    [486.218, 109.415],
    [503.097, 114.454],
    [519.714, 120.090],
    [533.680, 127.609],
    [571.937, 123.590],
    [585.702, 117.155],
    [602.344, 109.070],
    [620.077, 103.951],
    [633.964, 111.236],
    [554.931, 145.072],
    [554.589, 161.106],
    [554.658, 177.570],
    [554.777, 194.295],
    [532.717, 197.930],
    [543.637, 202.841],
    [555.652, 205.483],
    [565.441, 202.069],
    [576.368, 197.061],
    [487.474, 136.436],
    [499.184, 132.337],
    [513.781, 133.589],
    [527.594, 143.047],
    [513.422, 144.769],
    [499.117, 144.737],
    [579.876, 140.815],
    [590.901, 130.008],
    [605.648, 128.376],
    [618.343, 132.671],
    [606.771, 140.525],
    [593.466, 141.419],
    [519.040, 229.040],
    [536.292, 221.978],
    [547.001, 221.192],
    [557.161, 224.381],
    [568.172, 219.826],
    [579.144, 222.233],
    [589.098, 224.410],
    [581.071, 239.804],
    [570.103, 251.962],
    [558.241, 254.844],
    [547.661, 254.621],
    [534.085, 247.772],
    [524.758, 230.477],
    [547.684, 231.663],
    [557.304, 230.805],
    [568.172, 229.159],
    [585.417, 225.992],
    [569.211, 237.777],
    [557.473, 240.542],
    [547.989, 240.014],
]


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


def test_calculate_d_outers():
    marks = np.asarray([TEST_MARKS])
    calculated = _calculate_d_outers(marks)
    original = np.asarray(
        [
            np.sqrt(
                (mark[LEFT_EYE_LEFT_LANDMARK][0] - mark[RIGHT_EYE_RIGHT_LANDMARK][0]) ** 2
                + (mark[LEFT_EYE_LEFT_LANDMARK][1] - mark[RIGHT_EYE_RIGHT_LANDMARK][1]) ** 2
            )
            for mark in marks
        ]
    )
    assert np.all(np.isclose(original, calculated))
