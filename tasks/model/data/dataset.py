import pyarrow.parquet as pq

from torch.utils.data import Dataset


class ParquetDataset(Dataset):

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.table = pq.read_table(self.path).to_pandas()

    def __len__(self):
        return self.table['glyphs'].shape[0]

    def __getitem__(self, index):
        return self.table['glyphs'][index], self.table['vectors'][index]
