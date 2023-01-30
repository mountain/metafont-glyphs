import glob
import os
import cv2
import pyarrow as pa
import pyarrow.parquet as pq


table = pq.read_table("data/dataset/validation.parquet").to_pandas()
for ix, glyph in enumerate(table['glyphs']):
    img = glyph.reshape((96, 96))
    cv2.imwrite("temp/%04d.png" % ix, img)

for ix, vector in enumerate(table['vectors']):
    vec = vector.reshape((-1, 2))
    with open("temp/%04d.csv" % ix, "w") as f:
        for points in vec:
            x, y = points
            if x < 0 and y < 0:
                f.write("%0.4f, %0.4f\n" % (x, y))
            elif x < 0 <= y:
                f.write("%0.4f, +%0.4f\n" % (x, y))
            elif x >= 0 > y:
                f.write("+%0.4f, %0.4f\n" % (x, y))
            else:
                f.write("+%0.4f, +%0.4f\n" % (x, y))
