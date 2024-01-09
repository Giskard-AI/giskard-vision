import numpy as np
import pytest

from giskard_vision.marks.facial_parts import FacialPart, FacialParts


def test_operator_or():
    eyes: FacialPart = FacialParts.LEFT_EYE.value | FacialParts.RIGHT_EYE.value
    assert eyes == FacialParts.EYES.value
    assert eyes.name != FacialParts.EYES.name


def test_operator_add():
    eyes: FacialPart = FacialParts.LEFT_EYE.value + FacialParts.RIGHT_EYE.value
    assert eyes == FacialParts.EYES.value
    assert eyes.name != FacialParts.EYES.name


def test_operator_substract():
    right_eye: FacialPart = FacialParts.EYES.value - FacialParts.LEFT_EYE.value
    assert right_eye == FacialParts.RIGHT_EYE.value
    assert right_eye.name != FacialParts.RIGHT_EYE.name


def test_operator_and():
    left_eye: FacialPart = FacialParts.LEFT_HALF.value & FacialParts.LEFT_EYE.value
    assert left_eye == FacialParts.LEFT_EYE.value
    assert left_eye.name != FacialParts.LEFT_EYE.name


def test_operator_xor():
    right_eye_nose: FacialPart = (FacialParts.EYES.value + FacialParts.NOSE.value) ^ FacialParts.LEFT_EYE.value
    assert right_eye_nose == (FacialParts.RIGHT_EYE.value + FacialParts.NOSE.value)


def test_operator_invert():
    assert ~(FacialParts.ENTIRE.value - FacialParts.LEFT_EYE.value) == FacialParts.LEFT_EYE.value


def test_sanity_checks():
    assert (
        (FacialParts.BOTTOM_HALF.value + FacialParts.UPPER_HALF.value)
        == FacialParts.ENTIRE.value
        == (FacialParts.RIGHT_HALF.value + FacialParts.LEFT_HALF.value)
    )
    assert (FacialParts.LEFT_EYEBROW.value + FacialParts.RIGHT_EYEBROW.value) == FacialParts.EYEBROWS.value
    assert (FacialParts.LEFT_EYE.value + FacialParts.RIGHT_EYE.value) == FacialParts.EYES.value
    assert (FacialParts.LEFT_CONTOUR.value + FacialParts.RIGHT_CONTOUR.value) == FacialParts.CONTOUR.value


def test_new_facial_part():
    fp = FacialPart.from_indices("example", 0, 1)
    other = np.zeros(68, dtype=bool)
    other[0] = True
    fp2 = FacialPart("another", other)
    assert fp == fp2
    assert np.array_equal(fp.idx, fp2.idx)


def test_bad_indices_should_raise():
    with pytest.raises(ValueError) as exc_info:
        FacialPart.from_indices("example", -1, 2)
        assert "Indices should be between 0 and 68" in str(exc_info)
    with pytest.raises(ValueError) as exc_info:
        FacialPart.from_indices("example", 0, 71)
        assert "Indices should be between 0 and 68" in str(exc_info)


def test_bad_operator_usage_should_raise():
    with pytest.raises(ValueError) as exc_info:
        FacialParts.LEFT_EYE.value | "toto"
        assert "Operator | is only implemented for FacialPart" in str(exc_info)
