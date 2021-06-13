#!/usr/bin/python

from distutils.core import setup, Extension

setup(
    name="autosat",
    version="0.1",
    packages=["autosat"],
    license="CC0",
    long_description="Library for making SAT instances",
    ext_modules=[
        Extension(
            "autosat_tseytin",
            sources=[
                "src/autosat/autosat_tseytin.cpp",
                "src/autosat/autosat_tseytin_wrap.cxx",
            ],
        ),
    ],
)

