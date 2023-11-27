from dataclasses import dataclass


@dataclass
class TestResult:
    name: str
    metric: float
    passed: bool
