# -*- mode: sh -*-
# build script for the project.
# Mingli Yuan <mingli.yuan@gmail.com>
#

mkdir -p temp/build # make directory `temp/build'
PYTHONPATH=./tasks/build python3 -m gen.bldgen # generating build.ninja

cd temp/build # entering directory `temp/build'
ninja -j 8 -f ../../tasks/build/build.ninja # bulilding

cd ../.. # leaving directory `temp/build'
