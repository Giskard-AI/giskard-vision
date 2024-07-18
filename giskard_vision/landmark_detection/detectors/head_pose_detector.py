from giskard_vision.core.dataloaders.wrappers import (
    CachedDataLoader,
    FilteredDataLoader,
)
from giskard_vision.landmark_detection.dataloaders.wrappers import HeadPoseDataLoader

from .base import LandmarkDetectionBaseDetector, Pose
from ...core.detectors.decorator import maybe_detector


@maybe_detector("headpose_landmark", tags=["vision", "face", "landmark", "filtered", "headpose"])
class HeadPoseDetectorLandmark(LandmarkDetectionBaseDetector):
    """
    Detector that evaluates models performance depending on the head position
    """

    issue_group = Pose

    selected_poses = {
        "roll": [(-180, -1e-5), (-1e-5, 1e-5), (1e-5, 180)],
        "yaw": [(-90, -45), (-45, -1e-5), (-1e-5, 1e-5), (1e-5, 45), (45, 90)],
        "pitch": [(-180, -1e-5), (-1e-5, 1e-5), (1e-5, 180)],
    }

    def get_dataloaders(self, dataset):
        cached_dl = CachedDataLoader(HeadPoseDataLoader(dataset), cache_size=None, cache_img=False, cache_labels=False)

        dls = []

        for key in self.selected_poses:
            for index in range(len(self.selected_poses[key])):
                try:
                    current_dl = FilteredDataLoader(cached_dl, self._get_predicate_function(key, index))
                    dls.append(current_dl)
                except ValueError:
                    pass

        return dls

    def _get_predicate_function(self, key, index):
        lower = self.selected_poses[key][index][0]
        upper = self.selected_poses[key][index][1]

        def predicate_function(elt):
            return elt[2].get_includes(key) > lower and elt[2].get_includes(key) < upper

        predicate_function.__name__ = f"head pose: {lower} < {key} < {upper}"
        return predicate_function
