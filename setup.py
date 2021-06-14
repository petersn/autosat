#!/usr/bin/python

from distutils.core import setup, Extension

setup(
    name="autosat",
    version="0.0.0",
    packages=["autosat"],
    license="CC0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=[
        Extension(
            "_autosat_tseytin",
            sources=[
                "src/autosat/autosat_tseytin.i",
                "src/autosat/autosat_tseytin.cpp",
            ],
            swig_opts=["-c++"],
        ),
    ],
)

