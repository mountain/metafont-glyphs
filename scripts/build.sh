mkdir -p temp
PYTHONPATH=./build python3 -m gen.bldgen
cd temp && ninja -j 8 -f ../build/build.ninja
