#!/bin/bash

#DOCKER=quay.io/pypa/manylinux_2_24_x86_64
#PLATFORM=manylinux_2_24_x86_64
DOCKER=quay.io/pypa/manylinux2014_x86_64
PLATFORM=manylinux2014_x86_64


sudo docker run -it -v `readlink -f .`:/autosat $DOCKER bash -c "cd /autosat/; time ./script_run_inside_docker.sh $PLATFORM"

