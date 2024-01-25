from math import inf

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

    selected_poses = {
        "roll": [(-inf, -1e-5), (-1e-5, 1e-5), (1e-5, inf)],
        "yaw": [(-90, -45), (-45, -1e-5), (-1e-5, 1e-5), (1e-5, 45), (45, 90)],
        "pitch": [(-inf, -1e-5), (-1e-5, 1e-5), (1e-5, inf)],
    }

    def get_dataloaders(self, dataset):
        cached_dl = CachedDataLoader(HeadPoseDataLoader(dataset), cache_size=None, cache_img=False, cache_marks=False)

        dls = []

        for key in self.selected_poses:
            for index in range(len(self.selected_poses[key])):
                try:
                    current_dl = FilteredDataLoader(cached_dl, self._map_pose(key, index))
                    dls.append(current_dl)
                except ValueError:
                    pass

        return dls

    def _map_pose(self, key, index):
        lower = self.selected_poses[key][index][0]
        upper = self.selected_poses[key][index][1]

        def current_map(elt):
            return elt[2]["headPose"][key] > lower and elt[2]["headPose"][key] < upper

        current_map.__name__ = f"head pose: {lower} < {key} < {upper}"
        return current_map
