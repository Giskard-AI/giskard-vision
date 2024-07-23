from typing import List, Union


class GiskardImportError(ImportError):
    def __init__(self, missing_packages: Union[List[str], str]) -> None:
        super().__init__()

        if isinstance(missing_packages, list):
            mp_string = " ".join(missing_packages)
            self.msg = f"The '{missing_packages}' Python packages are not installed; please execute 'pip install {mp_string}' to obtain it."
        elif isinstance(missing_packages, str):
            self.msg = f"The '{missing_packages}' Python package is not installed; please execute 'pip install {missing_packages}' to obtain it."
        else:
            raise ValueError(
                f"{self.__class__.__name__}: takes only a list of strings, or a single string, instead {type(missing_packages)} was given"
            )


class GiskardError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
