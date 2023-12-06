from dataclasses import dataclass


@dataclass
class TestResult:
    name: str
    description: str
    metric: float
    passed: bool
