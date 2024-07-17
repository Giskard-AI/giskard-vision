from typing import Any, Dict, List, Optional


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

    data: Dict[str, Any]
    categories: Optional[List[str]]

    def __init__(self, data: Dict[str, Any], categories: Optional[List[str]] = None):
        """
        Constructs all the necessary attributes for the MetaData object.

        Args:
            data (Dict[str, Any]): The metadata as a dictionary.
            categories (Optional[List[str]]): The categories of the metadata, defaults to None.
        """
        self._data = data
        self._categories = categories

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
            Dict[str, Any]: A dictionary of non-iterable items from the metadata.
        """
        return list({k: v for k, v in self.data.items() if self.is_scannable(v)}.keys())

    def get_categories(self) -> Optional[List[str]]:
        """
        Returns the categories of the metadata.

        Returns:
            Optional[List[str]]: The categories of the metadata, or None if no categories were provided.
        """
        return self.categories
