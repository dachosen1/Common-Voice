from model.config.config import PACKAGE_ROOT

VERSION_PATH = PACKAGE_ROOT / "VERSION"

with open(VERSION_PATH, "r") as version_file:
    __version__ = version_file.read().strip()
