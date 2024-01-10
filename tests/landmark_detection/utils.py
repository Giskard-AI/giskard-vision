import shutil
from pathlib import Path
from urllib.request import urlretrieve


def fetch_remote(url: str, file: Path) -> None:
    if not file.parent.exists():
        file.parent.mkdir(parents=True, exist_ok=True)

    if not file.exists():
        urlretrieve(url, file)


def ungzip(file: Path, output_dir: Path):
    shutil.unpack_archive(
        filename=file,
        extract_dir=output_dir,
    )
