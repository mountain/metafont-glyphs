# -*- mode: sh -*-
# fontg script for the project.
# Mingli Yuan <mingli.yuan@gmail.com>
#

mkdir -p temp/build

PYTHONPATH=./tasks/build python3 -m gen.metagen "$1"
