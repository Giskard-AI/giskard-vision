class GiskardImportError(ImportError):
    def __init__(self, missing_package: str) -> None:
        self.msg = f"The '{missing_package}' Python package is not installed; please execute 'pip install {missing_package}' to obtain it."
