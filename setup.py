#!/usr/bin/python

from distutils.core import setup, Extension

setup(
    name="autosat",
    version="0.4.0",
    packages=["autosat"],
    license="CC0",
    long_description="Library for making SAT instances",
    ext_modules=[
        Extension(
            "_autosat_tseytin",
            sources=[
				"src/autosat/autosat_tseytin.i",
                "src/autosat/autosat_tseytin.cpp",
                #"src/autosat/autosat_tseytin_wrap.cxx",
            ],
			swig_opts=["-c++"],
        ),
    ],
)

