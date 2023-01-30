import glob
import os
import cv2
import pyarrow as pa
import pyarrow.parquet as pq


table = pq.read_table("data/dataset/validation.parquet").to_pandas()
for ix, glyph in enumerate(table['glyphs']):
    img = glyph.reshape((96, 96))
    cv2.imwrite("temp/%04d.png" % ix, img)

