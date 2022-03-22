#!/bin/bash

PLATFORM=$1

echo "Targeting platform: $PLATFORM"

time python3.6 -m build
time python3.7 -m build
time python3.8 -m build
time python3.9 -m build
time python3.10 -m build
rm dist/autosat*.tar.gz

echo "Running auditwheel repair to adjust to platform: $PLATFORM"

for WHEEL in dist/*.whl; do
	echo "Processing: $WHEEL"
	time auditwheel repair $WHEEL --plat $PLATFORM -w dist/
done

