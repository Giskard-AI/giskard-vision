from dataclasses import dataclass


@dataclass(frozen=True)
class IssueGroup:
    name: str
    description: str


EthicalIssueMeta = IssueGroup(
    "Ethical",
    description="The data are filtered by metadata like age, facial hair, or gender to detect ethical biases.",
)
PerformanceIssueMeta = IssueGroup(
    "Performance",
    description="The data are filtered by metadata like emotion, head pose, or exposure value to detect performance issues.",
)
AttributesIssueMeta = IssueGroup(
    "Attributes",
    description="The data are filtered by the image attributes like width, height, or brightness value to detect issues.",
)
Robustness = IssueGroup(
    "Robustness",
    description="Images from the dataset are blurred, recolored and resized to test the robustness of the model to transformations.",
)
