ninja -f ../build/build.ninja -t clean
cd ./temp && rm *.aux && rm *.tfm && cd ..
cd ./metafont && rm *.mf && cd ..
cd ./vector && rm *.csv && cd ..
cd ./glyph && rm *.png && cd ..
