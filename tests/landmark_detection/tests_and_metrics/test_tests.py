from giskard_vision.landmark_detection.dataloaders.wrappers import CroppedDataLoader
from giskard_vision.landmark_detection.marks.facial_parts import FacialParts
from giskard_vision.landmark_detection.tests.base import Test, TestDiff
from giskard_vision.landmark_detection.tests.performance import NMEMean


def test_tests_on_cropped_dl(opencv_model, dataset_300w):
    fp = FacialParts.LEFT_HALF.value
    dl = CroppedDataLoader(dataset_300w, part=fp)

    for test in [Test, TestDiff]:
        kwargs = {"model": opencv_model, "dataloader": dl, "facial_part": fp}
        if test == TestDiff:
            kwargs["dataloader_ref"] = dataset_300w

        test1 = test(metric=NMEMean, threshold=1).run(**kwargs)
        kwargs.pop("facial_part")
        test2 = test(metric=NMEMean, threshold=1).run(**kwargs)

        assert test1.metric_value == test2.metric_value
