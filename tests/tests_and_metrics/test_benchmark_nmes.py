import numpy as np
import pytest

from loreal_poc.tests.performance import NMEs


@pytest.mark.parametrize(
    "model_name, dataset_name, benchmark",
    [
        ("opencv_model", "dataset_300w", 0.04136279942306024),
        ("face_alignment_model", "dataset_300w", 0.06233510979950631),
    ],
)
def test_metric(model_name, dataset_name, benchmark, request):
    model = request.getfixturevalue(model_name)
    dataset = request.getfixturevalue(dataset_name)
    predictions = model.predict(dataset)
    nmes = NMEs.get(predictions, dataset.all_marks)
    dataset_nmes = np.nanmean(nmes)
    assert np.isclose(benchmark, dataset_nmes)
