import glob
import os
import cv2
import pyarrow as pa
import pyarrow.parquet as pq

flist = []
for gpath in sorted(glob.glob("glyph/*.png")):
    vpath = gpath.replace("glyph/", "vector/").replace(".png", ".csv")
    if not os.path.exists(vpath):
        continue
    flist.append(gpath.replace("glyph/", "").replace(".png", ""))
    if len(flist) == 10000:
        break

glyphs = []
vectors = []
for fname in sorted(flist):
    print(fname)
    glyph = cv2.imread("glyph/%s.png" % fname, cv2.IMREAD_GRAYSCALE).flatten()
    vector = []
    for line in open("vector/%s.csv" % fname).readlines():
        x, y = line.split(",")
        vector.append(float(x))
        vector.append(float(y))
    glyphs.append(glyph)
    vectors.append(vector)

table = pa.table({
    "vectors": pa.array(vectors[:1000], type=pa.list_(pa.float32())),
    "glyphs": pa.array(glyphs[:1000], type=pa.list_(pa.uint8(), 96 * 96))
})
pq.write_table(table, "validation.parquet")

table = pa.table({
    "vectors": pa.array(vectors[1000:4000], type=pa.list_(pa.float32())),
    "glyphs": pa.array(glyphs[1000:4000], type=pa.list_(pa.uint8(), 96 * 96))
})
pq.write_table(table, "test.parquet")

table = pa.table({
    "vectors": pa.array(vectors[4000:], type=pa.list_(pa.float32())),
    "glyphs": pa.array(glyphs[4000:], type=pa.list_(pa.uint8(), 96 * 96))
})
pq.write_table(table, "train.parquet")
