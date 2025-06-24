#!/usr/bin/env python
"""Setup script for pulsating-star-analysis package."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="pulsating-star-analysis",
    version="0.1.0",
    author="ligne-de-fuite-0",
    author_email="your.email@example.com",
    description="A Python toolkit for pulsating star lightcurve analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ligne-de-fuite-0/pulsating-star-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    keywords="astronomy, variable stars, lightcurves, machine learning, anomaly detection",
    project_urls={
        "Bug Reports": "https://github.com/ligne-de-fuite-0/pulsating-star-analysis/issues",
        "Source": "https://github.com/ligne-de-fuite-0/pulsating-star-analysis",
        "Documentation": "https://pulsating-star-analysis.readthedocs.io/",
    },
)