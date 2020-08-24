import os

from commonvoice.api.config import  PACKAGE_ROOT

with open(os.path.join(str(PACKAGE_ROOT), 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
