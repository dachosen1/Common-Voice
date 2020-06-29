import os


with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
) as version_file:
    __version__ = version_file.read().strip()