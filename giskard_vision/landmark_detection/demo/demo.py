from giskard_vision.landmark_detection.dataloaders.loaders import (
    DataLoader300W,
    DataLoaderFFHQ,
)


def get_300W():
    return DataLoader300W(dir_path="300W")


def get_ffhq():
    return DataLoaderFFHQ("../datasets/ffhq")
