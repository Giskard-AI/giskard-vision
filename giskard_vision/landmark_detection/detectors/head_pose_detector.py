from giskard_vision.landmark_detection.dataloaders.wrappers import (
    CachedDataLoader,
    FilteredDataLoader,
    HeadPoseDataLoader,
)

from .base import LandmarkDetectionBaseDetector
from .decorator import maybe_detector


@maybe_detector("headpose_landmark", tags=["vision", "face", "landmark", "filtered", "headpose"])
class HeadPoseDetectorLandmark(LandmarkDetectionBaseDetector):
    """
    Detector that evaluates models performance depending on the head position
    """

    group: str = "Head Pose"

    def get_dataloaders(self, dataset):
        cached_dl = CachedDataLoader(HeadPoseDataLoader(dataset), cache_size=None, cache_img=False, cache_marks=False)

        head_poses = [("positive roll", self._positive_roll), ("negative roll", self._negative_roll)]
        dls = []

        for hp in head_poses:
            current_dl = FilteredDataLoader(cached_dl, hp[1])
            dls.append(current_dl)

        return dls

    def _positive_roll(self, elt):
        return elt[2]["headPose"]["roll"] > 0

    def _negative_roll(self, elt):
        return elt[2]["headPose"]["roll"] < 0
