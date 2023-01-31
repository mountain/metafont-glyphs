import numpy as np
import pyarrow.parquet as pq

from torch.utils.data import Dataset


maxlen = 100


class ParquetDataset(Dataset):

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.table = pq.read_table(self.path).to_pandas()

    def __len__(self):
        return self.table['glyphs'].shape[0]

    def __getitem__(self, index):
        glyphs = np.array(list(self.table['glyphs'][index]), dtype=np.float32)
        vectors = np.array(list(self.table['vectors'][index]), dtype=np.float32)
        leng = vectors.shape[0]
        if leng < maxlen:
            aligned = np.pad(vectors, (0, maxlen - leng), 'constant', constant_values=-1)
        else:
            aligned = vectors[:maxlen]
        return glyphs, aligned
