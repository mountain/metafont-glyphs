# -*- mode: sh -*-
# mkds script for the project.
# Mingli Yuan <mingli.yuan@gmail.com>
#

mkdir -p data/dataset

PYTHONPATH=./tasks/dataset python3 -m ds.mkds "$1"
