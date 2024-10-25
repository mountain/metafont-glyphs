import numpy as np
import pyarrow.parquet as pq

from torch.utils.data import Dataset, DataLoader

STARTER = -1.0
maxlen = 80


class ParquetDataset(Dataset):

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.table = pq.read_table(self.path).to_pandas()

    def __len__(self):
        return self.table['glyphs'].shape[0]

    def __getitem__(self, index):
        glyphs = np.array(list(self.table['glyphs'][index]), dtype=np.float32) / 255
        vectors = np.array(list(self.table['vectors'][index]), dtype=np.float32)
        return glyphs, vectors


dltrain = DataLoader(ParquetDataset("../../data/dataset/train.parquet"), batch_size=1, num_workers=1, shuffle=False, drop_last=False, pin_memory=False, prefetch_factor=2, persistent_workers=True)
dlvalid = DataLoader(ParquetDataset("../../data/dataset/validation.parquet"), batch_size=1, num_workers=1, shuffle=False, drop_last=False, pin_memory=False, prefetch_factor=2, persistent_workers=True)
dltest = DataLoader(ParquetDataset("../../data/dataset/test.parquet"), batch_size=1, num_workers=1, shuffle=False, drop_last=False, pin_memory=False, prefetch_factor=2, persistent_workers=True)


def build_vocab():
    vocab = {
        STARTER: 0 # special token for start
    }
    for dl in [dltrain, dlvalid, dltest]:
        for item in dl:
            _, strokes = item
            for code in strokes.flattern():
                if code not in vocab:
                    # keep 4 points of precision
                    code = round(code, 4)
                    vocab[code] = len(vocab)

    return vocab


VOCAB2ID = build_vocab()
ID2VOCAB = {v: k for k, v in VOCAB2ID.items()}

print("size of vocab:", len(VOCAB2ID))


class VocabDataset(Dataset):

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.table = pq.read_table(self.path).to_pandas()

    def __len__(self):
        return self.table['glyphs'].shape[0]

    def __getitem__(self, index):
        glyphs = np.array(list(self.table['glyphs'][index]), dtype=np.float32) / 255
        vectors = np.array(list(self.table['vectors'][index]), dtype=np.float32)

        leng = vectors.shape[0]
        ids = np.zeros([leng+1], dtype=np.int64)
        ids[0] = VOCAB2ID[STARTER]
        for i in range(len(vectors)):
            ids[i+1] = VOCAB2ID[vectors[i]]

        if leng < maxlen:
            aligned = np.pad(ids, (0, maxlen - leng), 'constant', constant_values=VOCAB2ID[STARTER])
        else:
            aligned = ids[:maxlen]

        return glyphs, aligned
