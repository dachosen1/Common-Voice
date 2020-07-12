#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import setup

# Package meta-data.
NAME = "Common voice classifier"

DESCRIPTION = "Approach to classify person age, gender country of origin based on voice"

long_description = (
    "This modules uses data from Common voice which is a crowdsourcing project started "
    "by Mozilla to create a free database for speech recognition software. The project is supported by "
    "volunteers who record sample sentences with a microphone and review recordings of other users."
    " The transcribed sentences will be collected in a voice database available under the public domain "
    "license CC0. This license ensures that developers can use the database for voice-to-text "
    "applications without restrictions or costs. Common Voice appeared as a response to the language "
    "assistants of large companies such as Amazon Echo, Siri or Google Assistant. "
    ""
    ""
    "In this modules I will provide an approach to classify in real time a person voice into the three "
    "broad categories of the commonvoice-voice-voice data set"
)

EMAIL = "anderson.nelson1@gmail.com"
AUTHOR = "Anderson Nelson"
REQUIRES_PYTHON = '>=3.5.4'

URL = 'https://github.com/dachosen1/Common-Voice'


# What packages are required for this module to be executed?
def list_reqs(name="requirements.txt"):
    with open(name) as fd:
        return fd.read().splitlines()


# Load the package's __version__.py module as a dictionary.

PACKAGE_ROOT = os.path.join(os.getcwd())
about = {}
with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as f:
    _version = f.read().strip()
    about['__version__'] = _version

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    install_requires=list_reqs(),
    include_package_data=True,
    license="MIT",
)


