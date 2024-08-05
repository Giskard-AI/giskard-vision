from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL.Image import Image as PILImage

from giskard_vision.core.detectors.base import IssueGroup


class MetaData:
    """
    A class to represent metadata.

    Attributes:
        data (Dict[str, Any]): The metadata as a dictionary.
        categories (Optional[List[str]]): The categories of the metadata.

    Methods:
        get() -> Dict[str, Any]: Returns the metadata.
        is_scannable(value: Any) -> bool: Checks if a value is non-iterable.
        get_scannable() -> Dict[str, Any]: Returns only the non-iterable items from the metadata.
        get_categories() -> Optional[List[str]]: Returns the categories of the metadata.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        categories: Optional[List[str]] = None,
        issue_groups: Optional[Dict[str, IssueGroup]] = None,
    ):
        """
        Constructs all the necessary attributes for the MetaData object.

        Args:
            data (Dict[str, Any]): The metadata as a dictionary.
            categories (Optional[List[str]]): The categories of the metadata, defaults to None.
        """
        self._data = data
        self._categories = categories
        self._issue_groups = issue_groups

    @property
    def data(self) -> Dict[str, Any]:
        """
        Returns the metadata.

        Returns:
            Dict[str, Any]: The metadata.
        """
        return self._data

    @property
    def categories(self) -> Optional[List[str]]:
        """
        Returns the categorical keys.

        Returns:
            Optional[List[str]]: The categorical keys.
        """
        return self._categories

    @property
    def issue_groups(self) -> Optional[Dict[str, IssueGroup]]:
        """
        Returns the IssueGroup map

        Returns:
            Optional[Dict[str, IssueGroup]]: IssueGroups of the metadata.
        """
        return self._issue_groups

    def issue_group(self, key: str) -> IssueGroup:
        """
        Returns the IssueGroup for a specific metadata.

        Returns:
            IssueGroup: IssueGroup of the metadata.
        """
        return self._issue_groups[key] if (self._issue_groups and key in self._issue_groups) else None

    def get(self, key: str) -> Any:
        """
        Returns the value for specifc key

        Returns:
            Any: The metadata value.
        """
        if key in self.data:
            return self.data[key]
        raise KeyError(f"Key '{key}' not found in the metadata")

    def get_includes(self, substr: str) -> Any:
        """
        Returns the value for a specific key if the substring is in the key.

        Args:
            substr (str): The substring to search for in the keys.

        Returns:
            Any: The metadata value for the key containing the substring.

        Raises:
            KeyError: If no keys containing the substring are found in the metadata.
            ValueError: If multiple keys containing the substring are found in the metadata.
        """
        # Collect all keys that contain the substring
        matching_keys = [key for key in self.data if substr in key]

        if not matching_keys:
            raise KeyError(f"No keys containing '{substr}' found in the metadata")
        elif len(matching_keys) > 1:
            raise ValueError(f"Multiple keys containing '{substr}' found in the metadata: {matching_keys}")
        else:
            # Only one key matched
            return self.data[matching_keys[0]]

    @staticmethod
    def is_scannable(value: Any) -> bool:
        """
        Checks if a value is non-iterable.

        Args:
            value (Any): The value to check.

        Returns:
            bool: True if the value is non-iterable, False otherwise.
        """
        return not isinstance(value, (list, tuple, dict, set))

    def get_scannables(self) -> List[str]:
        """
        Returns only the non-iterable items from the metadata.

        Returns:
            List[str]: A list of non-iterable items from the metadata.
        """
        return list({k: v for k, v in self.data.items() if self.is_scannable(v)}.keys())

    def get_categories(self) -> Optional[List[str]]:
        """
        Returns the categories of the metadata.

        Returns:
            Optional[List[str]]: The categories of the metadata, or None if no categories were provided.
        """
        return self.categories


def get_image_size(image: np.ndarray) -> Tuple[int, int]:
    """
    Utitlity to create metadata with image size.

    Args:
        image (np.ndarray): The numpy ndarray representation of an image.

    Returns:
        Tuple[int, int]: The image size, width and height.
    """
    return image.shape[:2]


def get_image_channel_number(image: np.ndarray) -> int:
    """
    Utitlity to create metadata with image channel number.

    Args:
        image (np.ndarray): The numpy ndarray representation of an image.

    Returns:
        int: The image channel number.
    """
    shape = image.shape
    return shape[2] if len(shape) > 2 else 1


def get_pil_image_depth(image: PILImage) -> int:
    """
    Utitlity to create metadata with image depth.

    Args:
        image (PILImage): The PIL Image object.

    Returns:
        int: The image depth.
    """
    mode = image.mode
    if mode == "1":
        return 1
    elif mode == "L":
        return 8
    elif mode == "P":
        return 8
    elif mode == "RGB":
        return 24
    elif mode == "RGBA":
        return 32
    elif mode == "CMYK":
        return 32
    elif mode == "YCbCr":
        return 24
    elif mode == "LAB":
        return 24
    elif mode == "HSV":
        return 24
    elif mode == "I":
        return 32
    elif mode == "F":
        return 32
    return 0


def get_brightness(image: np.ndarray) -> float:
    """
    Utitlity to create metadata with image brightness.

    Args:
        image (np.ndarray): The numpy ndarray representation of an image.

    Returns:
        float: The image brightness normalized to 1.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2]) / 255


def get_entropy(image: np.ndarray) -> float:
    """
    Utitlity to create metadata with image entropy.

    Args:
        image (np.ndarray): The numpy ndarray representation of an image.

    Returns:
        float: The image entropy.
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist /= hist.sum()
    # Add eps to avoid log(0)
    return -np.sum(hist * np.log2(hist + np.finfo(float).eps))
