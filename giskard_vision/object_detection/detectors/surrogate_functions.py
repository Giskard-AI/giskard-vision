import numpy as np

from giskard_vision.core.detectors.metadata_scan_detector import Surrogate


@staticmethod
def center_mass_x(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    center_x = (x_min + x_max) / 2
    return center_x / image.shape[0]


SurrogateCenterMassX = Surrogate("center_mass_x", center_mass_x)


@staticmethod
def center_mass_y(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    center_y = (y_min + y_max) / 2
    return center_y / image.shape[1]


SurrogateCenterMassY = Surrogate("center_mass_y", center_mass_y)


@staticmethod
def area(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    area = (x_max - x_min) * (y_max - y_min)
    return area / (image.shape[0] * image.shape[1])


SurrogateArea = Surrogate("area", area)


@staticmethod
def aspect_ratio(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    width = x_max - x_min
    height = y_max - y_min
    return width / height


SurrogateAspectRatio = Surrogate("aspect_ratio", aspect_ratio)


@staticmethod
def normalized_width(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    width = x_max - x_min
    normalized_width = width / image.shape[1]
    return normalized_width


SurrogateNormalizedWidth = Surrogate("normalized_width", normalized_width)


@staticmethod
def normalized_height(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    height = y_max - y_min
    normalized_height = height / image.shape[0]
    return normalized_height


SurrogateNormalizedHeight = Surrogate("normalized_height", normalized_height)


@staticmethod
def normalized_perimeter(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    width = x_max - x_min
    height = y_max - y_min
    perimeter = 2 * (width + height)
    normalized_perimeter = perimeter / (2 * (image.shape[0] + image.shape[1]))
    return normalized_perimeter


SurrogateNormalizedPerimeter = Surrogate("normalized_perimeter", normalized_perimeter)


@staticmethod
def relative_top_left_x(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    relative_x = x_min / float(image.shape[0])
    return relative_x


SurrogateRelativeTopLeftX = Surrogate("relative_top_left_x", relative_top_left_x)


@staticmethod
def relative_top_left_y(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    relative_y = y_min / float(image.shape[1])
    return relative_y


SurrogateRelativeTopLeftY = Surrogate("relative_top_left_y", relative_top_left_y)


@staticmethod
def relative_bottom_right_x(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    relative_x = x_max / float(image.shape[0])
    return relative_x


SurrogateRelativeBottomRightX = Surrogate("relative_bottom_right_x", relative_bottom_right_x)


@staticmethod
def relative_bottom_right_y(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    relative_y = y_max / float(image.shape[1])
    return relative_y


SurrogateRelativeBottomRightY = Surrogate("relative_bottom_right_y", relative_bottom_right_y)


@staticmethod
def distance_from_center(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    image_center_x = image.shape[1] / 2
    image_center_y = image.shape[0] / 2
    distance = np.sqrt((center_x - image_center_x) ** 2 + (center_y - image_center_y) ** 2)
    return distance


SurrogateDistanceFromCenter = Surrogate("distance_from_center", distance_from_center)


@staticmethod
def mean_intensity(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    roi = image[int(y_min) : int(y_max), int(x_min) : int(x_max)]
    mean_intensity = roi.mean()
    return mean_intensity


SurrogateMeanIntensity = Surrogate("mean_intensity", mean_intensity)


@staticmethod
def std_intensity(result, image):
    x_min, y_min, x_max, y_max = result[0]["boxes"]
    roi = image[int(y_min) : int(y_max), int(x_min) : int(x_max)]
    std_intensity = roi.std()
    return std_intensity


SurrogateStdIntensity = Surrogate("std_intensity", std_intensity)
