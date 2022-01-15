#!/usr/bin/env python

import setuptools

setuptools.setup(
    name="wordlebot",
    version="0.0",
    install_requires=[],
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    entry_points={
        "botfights_sdk.wordle": [
            "wordlebot = wordlebot.guessing:Guesser",
        ],
    },
)
