from typing import Dict, Optional, Tuple, Union

import numpy as np

from giskard_vision.core.dataloaders.base import DataLoaderWrapper
from giskard_vision.core.dataloaders.meta import MetaData
from giskard_vision.core.dataloaders.wrappers import ResizedDataLoaderBase
from giskard_vision.landmark_detection.marks.facial_parts import FacialPart
from giskard_vision.landmark_detection.transformations import (
    crop_image_from_mark,
    crop_mark,
    resize_marks,
)
from giskard_vision.utils.errors import GiskardImportError

from .base import DataIteratorBase


class CroppedDataLoader(DataLoaderWrapper):
    """Wrapper class for a DataIteratorBase, providing cropped images based on facial landmarks.

    Args:
        dataloader (DataIteratorBase): The data loader to be wrapped.
        part (FacialPart): The facial part to be cropped.
        margins (Union[Tuple[float, float], float]): Margins for cropping. Can be a tuple or a single value.

    Returns:
        CroppedDataLoader: Cropped data loader instance.
    """

    def __init__(
        self,
        dataloader: DataIteratorBase,
        part: FacialPart,
        margins: Union[Tuple[float, float], float] = (0, 0),
    ) -> None:
        """
        Initializes the CroppedDataLoader.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
            part (FacialPart): The facial part to be cropped.
            margins (Union[Tuple[float, float], float]): Margins for cropping. Can be a tuple or a single value.
        """
        super().__init__(dataloader)
        self._part = part
        self._margins = margins

    @property
    def name(self) -> str:
        """
        Gets the name of the cropped data loader.

        Returns:
            str: The name of the cropped data loader.
        """
        return f"cropped on {self._part.name}"  # f"{self._wrapped_dataloader.name} cropped on {self._part.name}"

    @property
    def facial_part(self) -> FacialPart:
        """
        Gets the facial_part used for the copping.

        Returns:
            FacialPart: The name of the cropped data loader.
        """
        return self._part

    def get_image(self, idx: int) -> np.ndarray:
        """
        Gets a cropped image based on facial landmarks.

        Args:
            idx (int): Index of the data.

        Returns:
            np.ndarray: Cropped image data.
        """
        image = super().get_image(idx)
        h, w, _ = image.shape
        margins = np.array([w, h]) * self._margins
        marks = crop_mark(self.get_labels_with_default(idx), self._part)
        return crop_image_from_mark(image, marks, margins)


class ResizedDataLoader(ResizedDataLoaderBase):
    """Wrapper class for a DataIteratorBase, providing resized images and landmarks.

    Args:
        dataloader (DataIteratorBase): The data loader to be wrapped.
        scales (Union[Tuple[float, float], float]): Scaling factors for resizing. Can be a tuple or a single value.
        absolute_scales (Optional[bool]): Flag indicating whether the scales are absolute or relative.

    Returns:
        ResizedDataLoader: Resized data loader instance.
    """

    def resize_labels(self, labels: np.ndarray, scales: Union[Tuple[float, float], float]) -> np.ndarray:
        return resize_marks(labels, scales)


class HeadPoseDataLoader(DataLoaderWrapper):
    """Wrapper class for a DataIteratorBase, providing head pose estimation using the SixDRepNet model.

    Args:
        dataloader (DataIteratorBase): The data loader to be wrapped.
        gpu_id (int, optional): GPU ID for model execution. Defaults to -1 (CPU).

    Raises:
        GiskardImportError: Error to signal a missing package.

    Returns:
        HeadPoseDataLoader: Head pose data loader instance.

    License information:
        Check giskard-vision/licenses
    """

    def __init__(self, dataloader: DataIteratorBase, gpu_id: int = -1) -> None:
        """
        Initializes the HeadPoseDataLoader.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
            gpu_id (int, optional): GPU ID for model execution. Defaults to -1 (CPU).
        """
        try:
            from sixdrepnet import SixDRepNet
        except ImportError as e:
            raise GiskardImportError("sixdrepnet") from e

        super().__init__(dataloader)

        self.pose_detection_model = SixDRepNet(gpu_id=gpu_id)

    # @property
    # def name(self):
    #    """
    #    Gets the name of the head pose data loader.
    #
    #    Returns:
    #        str: The name of the head pose data loader.
    #    """
    #    return f"({self._wrapped_dataloader.name}) with head-pose"

    def get_meta(self, idx):
        """
        Gets (predicts) head pose metadata for the specified index.

        Args:
            idx (int): Index of the data.

        Returns:
            Types.meta: Head pose metadata including pitch, yaw, and roll.
        """
        pitch, yaw, roll = self.pose_detection_model.predict(self.get_image(idx))
        return MetaData(data={"pitch": pitch[0], "yaw": -yaw[0], "roll": roll[0]})


class EthnicityDataLoader(DataLoaderWrapper):
    """Wrapper class for a DataIteratorBase, providing ethnicity estimation using the DeepFace model.

    Args:
        dataloader (DataIteratorBase): The data loader to be wrapped.
        ethnicity_map (Optional[Dict]): Mapping of custom ethnicity labels to DeepFace ethnicity labels.

    Returns:
        EthnicityDataLoader: Ethnicity data loader instance.

    License information:
        Check giskard-vision/licenses
    """

    supported_ethnicities = [
        "indian",
        "asian",
        "latino hispanic",
        "middle eastern",
        "white",
    ]

    def __init__(self, dataloader: DataIteratorBase, ethnicity_map: Optional[Dict] = None) -> None:
        """
        Initializes the EthnicityDataLoader.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
            ethnicity_map (Optional[Dict]): Mapping of custom ethnicity labels to DeepFace ethnicity labels.
        """
        super().__init__(dataloader)

        if ethnicity_map is not None:
            keys_and_values = set(ethnicity_map.keys()).union(set(ethnicity_map.values()))
            if not keys_and_values.issubset(self.supported_ethnicities):
                raise ValueError(
                    f"Only the following ethnicities {self.supported_ethnicities} are estimated by DeepFace."
                )
            if set(ethnicity_map.keys()).issubset(ethnicity_map.values()):
                raise ValueError("Only one-to-one mapping is allowed in ethnicity_map.")
        self.ethnicity_map = ethnicity_map

    # @property
    # def name(self):
    #    """
    #    Gets the name of the ethnicity data loader.
    #
    #    Returns:
    #        str: The name of the ethnicity data loader.
    #    """
    #    return f"({self._wrapped_dataloader.name}) with ethnicity"

    def _map_ethnicities(self, ethnicities: Dict):
        """
        Maps custom ethnicities to DeepFace ethnicity labels.

        Args:
            ethnicities (Dict): Ethnicity labels.

        Returns:
            Dict: Mapped ethnicity labels.
        """
        # merging
        for k, v in self.ethnicity_map.items():
            ethnicities[v] += ethnicities[k]
        # purging
        [ethnicities.pop(k) for k in self.ethnicity_map.keys()]
        return ethnicities

    def get_meta(self, idx):
        """
        Gets ethnicity metadata for the specified index.

        Args:
            idx (int): Index of the data.

        Returns:
            Types.meta: Ethnicity metadata.
        """
        try:
            from deepface import DeepFace
        except ImportError as e:
            raise GiskardImportError("deepface") from e
        try:
            ethnicities = DeepFace.analyze(img_path=self.get_image(idx), actions=["race"])[0]["race"]
            ethnicities = self._map_ethnicities(ethnicities) if self.ethnicity_map else ethnicities
            return MetaData(data={"ethnicity": max(ethnicities, key=ethnicities.get)})
        except ValueError:
            return MetaData(data={"ethnicity": "unknown"})
