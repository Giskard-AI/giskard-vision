from typing import Union, List

class GiskardImportError(ImportError):
    def __init__(self, missing_packages: Union[List[str], str]) -> None:
        super().__init__()
        
        if isinstance(missing_packages, list):
            self.msg = f"The '{missing_packages}' Python packages are not installed; please execute 'pip install {" ".join(missing_packages)}' to obtain it."
        elif isinstance(missing_packages, str):
            self.msg = f"The '{missing_packages}' Python package is not installed; please execute 'pip install {missing_packages}' to obtain it."
        else:
            raise ValueError(f"{self.__class__.__name__}: takes only a list of strings, or a single string, instead {type(missing_packages)} was given")
