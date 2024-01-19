try:
    from giskard_vision.landmark_detection import detectors

    from . import landmark_detection

    __all__ = ["landmark_detection", "detectors"]
except (ImportError, ModuleNotFoundError):
    print("Please install giskard to use custom detectors")
    __all__ = ["landmark_detection"]
