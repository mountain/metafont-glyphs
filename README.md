# metafont-glyphs dataset

This repository contains a dataset of glyph images generated from random metafont scripts.

We have many variant characters from ancient literature, such as oracle-bone inscriptions, bronze inscriptions etc.
The main difficulty is that these characters are not standardized, and in order to digitize them, we need to collect a large number of glyph images,
and then use machine learning to generate metafont scripts from glyph images.
It is a different problem from OCR, which is focused on recognizing and transcribing written text from images.
The goal is to develop models that can generate accurate geometric descriptions of these glyphs, 
which can then be used in typesetting systems such as Metafont and Tex.

## gallery
<div>
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/01.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/02.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/03.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/04.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/05.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/06.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/07.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/08.png" width="96px">
</div>
<div>
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/09.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/10.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/11.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/12.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/13.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/14.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/15.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/16.png" width="96px">
</div>
<div>
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/17.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/18.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/19.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/20.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/21.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/22.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/23.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/24.png" width="96px">
</div>

## An example

The following glyph is generated from the following metafont script.

<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/00.png" width="96px">

```metafont
% file name: bcdylw.mf
% mode_setup;
% Define a random shape for the training corpus

beginchar("bcdylw",12pt#,12pt#,0);
  % Setup coordinates as an equation system
  x12 = 1 * w / 10;
  x14 = 1 * w / 10;
  x32 = 3 * w / 10;
  x34 = 3 * w / 10;
  x36 = 3 * w / 10;
  x38 = 3 * w / 10;
  x45 = 4 * w / 10;
  x68 = 6 * w / 10;

  y12 = 2 * w / 10;
  y14 = 4 * w / 10;
  y32 = 2 * w / 10;
  y34 = 4 * w / 10;
  y36 = 6 * w / 10;
  y38 = 8 * w / 10;
  y45 = 5 * w / 10;
  y68 = 8 * w / 10;

  % Draw the character curve
  % z1 is the same as (x1, y1)
  pickup pencircle xscaled 0.06w yscaled 0.02w rotated 243;
  draw z36..z38;
  draw z36..z38;
  draw z12..z14;
  draw z45..z68..z34;
  draw z32..z34;
endchar;

end
```

To facilitate machine learning, we also provide the following csv file for the sequence of coordinates of the control points.

```csv
-0.1666, +0.5750
-0.5000, +0.2497
-0.8333, +0.6750
+0.3000, +0.6000
+0.3000, +0.8000
-0.5000, -0.5000
+0.3000, +0.6000
+0.3000, +0.8000
-0.5000, -0.5000
+0.1000, +0.2000
+0.1000, +0.4000
-0.5000, -0.5000
+0.4000, +0.5000
+0.6000, +0.8000
+0.3000, +0.4000
-0.5000, -0.5000
+0.3000, +0.2000
+0.3000, +0.4000
-0.5000, -0.5000
```

Lines with negative coordinates are special
  * `-0.1666, +0.5750` means the xscale of the pen is 0.0575
  * `-0.5000, +0.2497` means the yscale of the pen is 0.02497
  * `-0.8333, +0.6750` means the rotation of the pen is 0.675 * 360 degrees, i.e. 243 degrees
  * `-0.5000, -0.5000` means the pen is lifted up, i.e. the pen is not drawing and hence the end of the curve

## How to use

### Install texlive

```bash
sudo apt install texlive-full # for Debian and Ubuntu
brew install texlive # for macOS
```

### Install python3, ninja and imagemagick

```bash
sudo apt install python3 # for Debian and Ubuntu
sudo apt install ninja-build # for Debian and Ubuntu
sudo apt install imagemagick # for Debian and Ubuntu
brew install python3 # for macOS
brew install ninja # for macOS
brew install imagemagick # for macOS
pip3 install -r requirements.txt # for any platform
```

### Fire the tests

```bash
git clone https://github.com/mountain/metafont-glyphs.git
cd metafont-glyphs
sh fontg.sh
sh build.sh
```





