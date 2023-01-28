mkdir -p temp
PYTHONPATH=./build python3 -m gen.metagen "$1"
