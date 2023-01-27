# metafont-glyphs dataset

This repository contains a dataset of glyph images generated from random metafont scripts.

## gallery
<div>
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/01.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/02.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/03.png" width="96px">
</div>
<div>
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/04.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/05.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/06.png" width="96px">
</div>
<div>
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/07.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/08.png" width="96px">
<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/09.png" width="96px">
</div>

## An example

The following glyph is generated from the following metafont script.

<img src="https://raw.githubusercontent.com/mountain/metafont-glyphs/main/demo/00.png" width="96px">

```metafont
% file name: xpogk.mf
% mode_setup;
% Define a random shape for the training corpus

beginchar("xpogk",12pt#,12pt#,0);
  % Setup coordinates as an equation system
  x11 = 1 * w / 10;
  x12 = 1 * w / 10;
  x13 = 1 * w / 10;
  x14 = 1 * w / 10;
  x15 = 1 * w / 10;
  x17 = 1 * w / 10;
  x18 = 1 * w / 10;
  x28 = 2 * w / 10;
  x32 = 3 * w / 10;
  x34 = 3 * w / 10;
  x35 = 3 * w / 10;
  x37 = 3 * w / 10;
  x43 = 4 * w / 10;
  x45 = 4 * w / 10;
  x46 = 4 * w / 10;
  x48 = 4 * w / 10;
  x51 = 5 * w / 10;
  x52 = 5 * w / 10;
  x55 = 5 * w / 10;
  x56 = 5 * w / 10;
  x62 = 6 * w / 10;
  x67 = 6 * w / 10;
  x68 = 6 * w / 10;
  x72 = 7 * w / 10;
  x73 = 7 * w / 10;
  x83 = 8 * w / 10;

  y11 = 1 * w / 10;
  y12 = 2 * w / 10;
  y13 = 3 * w / 10;
  y14 = 4 * w / 10;
  y15 = 5 * w / 10;
  y17 = 7 * w / 10;
  y18 = 8 * w / 10;
  y28 = 8 * w / 10;
  y32 = 2 * w / 10;
  y34 = 4 * w / 10;
  y35 = 5 * w / 10;
  y37 = 7 * w / 10;
  y43 = 3 * w / 10;
  y45 = 5 * w / 10;
  y46 = 6 * w / 10;
  y48 = 8 * w / 10;
  y51 = 1 * w / 10;
  y52 = 2 * w / 10;
  y55 = 5 * w / 10;
  y56 = 6 * w / 10;
  y62 = 2 * w / 10;
  y67 = 7 * w / 10;
  y68 = 8 * w / 10;
  y72 = 2 * w / 10;
  y73 = 3 * w / 10;
  y83 = 3 * w / 10;

  % Draw the character curve
  % z1 is the same as (x1, y1)
  pickup pencircle xscaled 0.08w yscaled 0.06w rotated 154;
  draw z62..z73;
  draw z52..z34;
  draw z15..z18;
  draw z55..z56..z46;
  draw z45..z48;
  draw z12..z13;
  draw z67..z68;
  draw z35..z37;
  draw z72..z43;
  draw z83..z73..z51;
  draw z11..z12;
  draw z34..z14;
  draw z17..z28..z18;
  draw z32..z35..z15;
  draw z17..z18;
endchar;

end
```
