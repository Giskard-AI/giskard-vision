from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw


def _add_marks(image_display, marks, color=None):
    marks = np.array(marks)
    if len(np.shape(marks)) == 2:
        marks = np.expand_dims(marks, axis=0)

    size = 5
    color = color if color is not None else tuple(np.random.choice(range(256), size=3))
    d = ImageDraw.Draw(image_display)
    for face_index in range(marks.shape[0]):
        for index, (x, y) in enumerate(marks[face_index]):
            d.ellipse([x - size / 2, y - size / 2, x + size // 2, y + size // 2], fill=color)
            # d.text((x, y), str(index), fill=(255, 0, 0))

    return image_display


def draw_marks(image: Image, list_of_marks: List, colors: Optional[List] = None):
    image_display = image.copy()
    for marks, color in zip(list_of_marks, colors or [None] * len(list_of_marks)):
        image_display = _add_marks(image_display, marks, color=color)
    return image_display
