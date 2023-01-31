import torch as th
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import data.dataset as ds

from abc import ABC
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader


if th.cuda.is_available():
    pass


class AbstractNet(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.lr = 0.02

    def make_plot(self, iname, ix, xs, ys):
        yd = ys[0, 0, :, :]
        glyph = yd.cpu().numpy().reshape((96, 96))
        cv2.imwrit('outputs/%s-%d' % (iname, yr), glyph)

    def forward(self, vector):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss(self, predict, target):
        raise NotImplementedError()

    def training_step(self, train_batch, batch_idx):
        xs, yt = train_batch
        ys = self.forward(*xs)
        lss = self.loss(ys, yt)
        self.log('train_loss', lss)
        return lss

    def validation_step(self, val_batch, batch_idx):
        xs, yt = val_batch
        ys = self.forward(*xs)
        lss = self.benchmark(ys, yt)
        self.log('val_loss', lss, prog_bar=True)
        if self.current_epoch % 100 == 19:
            self.make_plot('valid', batch_idx, xs, ys)
        return lss

    def test_step(self, test_batch, batch_idx):
        xs, yt = test_batch
        ys = self.forward(*xs)
        lss = self.benchmark(ys, yt)
        self.log('test_loss', lss)
        self.print_errors(ys, yt)
        self.make_plot('test', batch_idx, xs, ys)
        return lss

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(ds.ParquetDataset("data/dataset/train.parquet"), batch_size=16, num_workers=1, shuffle=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(ds.ParquetDataset("data/dataset/test.parquet"), batch_size=16, num_workers=1, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(ds.ParquetDataset("data/dataset/validation.parquet"), batch_size=16, num_workers=1, shuffle=True)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass


class Baseline(AbstractNet):
    def __init__(self):
        super().__init__()
        self.model_name = 'bs'

    def forward(self, vector):
        return pred, ylds

    def loss(self, predict, target):
        return self.benchmark(predict, target)


_model_ = Baseline

