from typing import List, Optional

import cv2
import numpy as np

from giskard_vision.landmark_detection.utils import GiskardImportError


def _add_marks(image_display, marks, color=None, with_text=False, square=None):
    try:
        from PIL import ImageDraw
    except ImportError as e:
        raise GiskardImportError("pillow") from e

    marks = np.array(marks)
    if len(np.shape(marks)) == 2:
        marks = np.expand_dims(marks, axis=0)

    size = 5
    color = color if color is not None else tuple(np.random.choice(range(256), size=3))

    d = ImageDraw.Draw(image_display)
    if square:
        d.rectangle(square, outline=color)
    for face_index in range(marks.shape[0]):
        for index, (x, y) in enumerate(marks[face_index]):
            if not np.isnan((x, y)).any():
                d.ellipse([x - size / 2, y - size / 2, x + size // 2, y + size // 2], fill=color)
                if with_text:
                    d.text((x, y), str(index), fill=(255, 0, 0))

    return image_display


def draw_marks(
    image: np.ndarray,
    list_of_marks: List,
    colors: Optional[List] = None,
    with_text: Optional[List] = None,
    squares=None,
):
    try:
        from PIL import Image
    except ImportError as e:
        raise GiskardImportError("pillow") from e

    if isinstance(image, np.ndarray):
        _image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_display = _image.copy()
    for marks, color, _with_text, square in zip(
        list_of_marks,
        colors or [None] * len(list_of_marks),
        with_text or [None] * len(list_of_marks),
        squares or [None] * len(list_of_marks),
    ):
        image_display = _add_marks(image_display, marks, color=color, with_text=_with_text, square=square)
    return image_display
