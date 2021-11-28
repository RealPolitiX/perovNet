#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='perovNet',
    version='0.1.0',
    description='Equivariant neural network for perovskites',
    author='',
    author_email='',
    url='https://github.com/RealPolitiX/perovNet',
    install_requires=['torch'],
    packages=find_packages(),
)

