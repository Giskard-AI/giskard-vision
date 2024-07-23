from typing import Any, Dict, List

import numpy as np


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "_", flat_np_array: bool = False
) -> Dict[str, Any]:
    """
    Flattens a nested dictionary.

    Args:
        d (Dict[str, Any]): The dictionary to flatten.
        parent_key (str): The base key string for the flattened keys.
        sep (str): The separator between keys.
        flat_np_array (bool): Flag to flatten numpy arrays. Default is `False`.

    Returns:
        Dict[str, Any]: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list) or (isinstance(v, np.ndarray) and flat_np_array):
            for i, item in enumerate(v):
                items.extend(_flatten_dict({f"{new_key}_{i}": item}, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_dict(
    d: Dict[str, Any], exclude_keys: List[str] = [], parent_key: str = "", sep: str = "_", flat_np_array: bool = False
) -> Dict[str, Any]:
    """
    Flattens a nested dictionary.

    Args:
        d (Dict[str, Any]): The dictionary to flatten.
        excludes (List[str]): Keys to exclude from the flattened dictionary.
        parent_key (str): The base key string for the flattened keys.
        sep (str): The separator between keys.
        flat_np_array (bool): Flag to flatten numpy arrays. Default is `False`.

    Returns:
        Dict[str, Any]: The flattened dictionary.
    """

    result = _flatten_dict(d, parent_key, sep, flat_np_array)

    for key in exclude_keys:
        if key in result:
            result.pop(key)

    return result
