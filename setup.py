#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="BLINK",
    version="0.1.0",
    description="BLINK: Better entity LINKing",
    url="",  # TODO
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    long_description=readme,
    long_description_content_type="text/markdown",
    setup_requires=["setuptools>=18.0",],
    install_requires=[
        "torch>=1.2.0",
        "pysolr>=3.8.1",
        "emoji>=0.5.3",
        "regex>=2019.8.19",
        "matplotlib>=3.1.0",
        "tqdm>=4.32.1",
        "nltk>=3.4.4",
        "numpy>=1.17.2",
        "segtok>=1.5.7",
        "flair>=0.4.3",
        "pytorch-transformers>=1.2.0",
        "colorama>=0.4.3",
        "termcolor>=1.1.0",
        "faiss-cpu>=1.6.1",
    ],
)
