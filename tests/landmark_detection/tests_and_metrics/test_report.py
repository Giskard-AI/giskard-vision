import tempfile

import numpy as np

from giskard_vision.landmark_detection.tests.report import Report


def test_report(opencv_model, dataset_300w):
    models = [opencv_model]
    dls = [dataset_300w]
    dl_ref = dataset_300w

    report = Report(models=models, dataloaders=dls)
    assert report.results[0]["test"] == "Test"
    assert not report.results[0]["passed"]
    assert np.allclose(report.results[0]["metric_value"], 0.04136279942)

    report.adjust_thresholds({0: 0.03})
    assert not report.results[0]["passed"]

    report2 = Report(models=models, dataloaders=dls, dataloader_ref=dl_ref)
    assert report2.results[0]["test"] == "TestDiff"

    with tempfile.NamedTemporaryFile() as f:
        report.to_json(filename=f.name)
        report2.to_json(filename=f.name)

    df = report.to_dataframe()
    df2 = report2.to_dataframe()
    assert len(df) == len(df2) == 1
