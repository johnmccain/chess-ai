# -*- coding: utf8 -*-
#
# This file were created by Python Boilerplate. Use Python Boilerplate to start
# simple, usable and best-practices compliant Python projects.
#
# Learn more about it at: http://github.com/fabiommendes/python-boilerplate/
#

import os

from setuptools import setup, find_packages

# Meta information
version = open('VERSION').read().strip()
dirname = os.path.dirname(__file__)

setup(
    # Basic info
    name='chess-ai',
    version=version,
    author='John McCain',
    author_email='johnm.freestate@gmail.com',

    # Packages and depencies
    package_dir={'': 'src'},
    packages=find_packages('src'),
    install_requires=[
        'jinja2',
        'invoke>=0.13',
        'unidecode',
        'six',
    ],
    extras_require={
        'dev': [
            'manuel',
            'pytest',
            'pytest-cov',
            'coverage',
            'mock',
        ],
    },
    # Other configurations
    zip_safe=False,
    platforms='any',
)
