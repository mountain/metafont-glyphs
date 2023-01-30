import glob
import os
import cv2
import pyarrow as pa
import pyarrow.parquet as pq


flist = []
for gpath in sorted(glob.glob("data/glyph/*/*/*.png")):
    vpath = gpath.replace("data/glyph/", "data/vector/").replace(".png", ".csv")
    if not os.path.exists(vpath):
        continue
    flist.append(gpath.replace("data/glyph/", "").replace(".png", ""))

dsize = len(flist)
vsize = int(dsize / 10)
tssize = int(dsize / 10) * 2
trsize = dsize - vsize - tssize
print("total data items:", dsize)
print("train size:", trsize)
print("valid size:", vsize)
print("test size:", tssize)

glyphs = []
vectors = []
for fname in sorted(flist):
    print(fname)
    glyph = cv2.imread("data/glyph/%s.png" % fname, cv2.IMREAD_GRAYSCALE).flatten()
    vector = []
    for line in open("data/vector/%s.csv" % fname).readlines():
        x, y = line.split(",")
        vector.append(float(x))
        vector.append(float(y))
    glyphs.append(glyph)
    vectors.append(vector)

table = pa.table({
    "vectors": pa.array(vectors[:vsize], type=pa.list_(pa.float32())),
    "glyphs": pa.array(glyphs[:vsize], type=pa.list_(pa.uint8(), 96 * 96))
})
pq.write_table(table, "data/dataset/validation.parquet")

table = pa.table({
    "vectors": pa.array(vectors[vsize:vsize + tssize], type=pa.list_(pa.float32())),
    "glyphs": pa.array(glyphs[vsize:vsize + tssize], type=pa.list_(pa.uint8(), 96 * 96))
})
pq.write_table(table, "data/dataset/test.parquet")

table = pa.table({
    "vectors": pa.array(vectors[vsize + tssize:], type=pa.list_(pa.float32())),
    "glyphs": pa.array(glyphs[vsize + tssize:], type=pa.list_(pa.uint8(), 96 * 96))
})
pq.write_table(table, "data/dataset/train.parquet")
