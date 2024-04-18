import os

from giskard_vision.landmark_detection.dataloaders.loaders import (
    DataLoader300W,
    DataLoaderFFHQ,
)


def get_300W():
    return DataLoader300W(os.path.join(os.path.dirname(__file__), "300W"))


def get_ffhq():
    return DataLoaderFFHQ(os.path.join(os.path.dirname(__file__), "ffhq"))
