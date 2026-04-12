#!/usr/bin/env python3
import os
import site
import sys
from setuptools import setup, find_packages

setup(
    name="kiwano",
    version="0.0.1",
    description="Speaker verification",
    author="Mickael Rouvier",
    author_email="mickael.rouvier@univ-avignon.fr",
    packages=find_packages(),
)
