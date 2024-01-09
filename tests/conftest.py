from glob import glob
from pathlib import Path

pytest_plugins = []
for f in glob("**/**/fixtures/**/*.py", recursive=True):
    path = Path(f)
    pytest_plugins.append(".".join([*path.parts[:-1], path.stem]))
