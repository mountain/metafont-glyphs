# -*- mode: sh -*-
# clean-all script for the project.
# Mingli Yuan <mingli.yuan@gmail.com>
#

cd temp/build
ninja -f ../build/build.ninja -t clean
cd ../..

cd data/metafont
rm b*.mf
rm c*.mf
rm d*.mf
rm f*.mf
rm g*.mf
rm h*.mf
rm j*.mf
rm k*.mf
rm l*.mf
rm m*.mf
rm n*.mf
rm p*.mf
rm q*.mf
rm r*.mf
rm s*.mf
rm t*.mf
rm v*.mf
rm w*.mf
rm x*.mf
rm y*.mf
rm z*.mf
cd ../..

cd data/vector
rm b*.csv
rm c*.csv
rm d*.csv
rm f*.csv
rm g*.csv
rm h*.csv
rm j*.csv
rm k*.csv
rm l*.csv
rm m*.csv
rm n*.csv
rm p*.csv
rm q*.csv
rm r*.csv
rm s*.csv
rm t*.csv
rm v*.csv
rm w*.csv
rm x*.csv
rm y*.csv
rm z*.csv
cd ../..

cd data/glyph
rm b*.png
rm c*.png
rm d*.png
rm f*.png
rm g*.png
rm h*.png
rm j*.png
rm k*.png
rm l*.png
rm m*.png
rm n*.png
rm p*.png
rm q*.png
rm r*.png
rm s*.png
rm t*.png
rm v*.png
rm w*.png
rm x*.png
rm y*.png
rm z*.png
cd ../..
