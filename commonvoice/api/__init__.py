from commonvoice.api.config import PACKAGE_ROOT
import os

with open(os.path.join(str(PACKAGE_ROOT), 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
