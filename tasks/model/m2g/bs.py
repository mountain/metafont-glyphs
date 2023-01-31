import torch as th
import cv2
import torch.nn as nn
import pytorch_lightning as pl
import data.dataset as ds

from abc import ABC
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from util.stroke import stroke


if th.cuda.is_available():
    pass


class AbstractNet(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.lr = 0.02

    def make_plot(self, iname, ix, xs, ys):
        yd = ys[0, 0, :, :]
        glyph = yd.cpu().numpy().reshape((96, 96))
        cv2.imwrit('outputs/%s-%d' % (iname, ix), glyph)

    def forward(self, vector):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss(self, predict, target):
        raise NotImplementedError()

    def training_step(self, train_batch, batch_idx):
        glyphs, vectors = train_batch
        images = self.forward(vectors).reshape(-1, 1, 96, 96)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        lss = self.benchmark(images, glyphs)
        self.log('train_loss', lss)
        return lss

    def validation_step(self, val_batch, batch_idx):
        glyphs, vectors = val_batch
        images = self.forward(vectors).reshape(-1, 1, 96, 96)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        lss = self.benchmark(images, glyphs)
        self.log('val_loss', lss, prog_bar=True)
        if self.current_epoch % 100 == 19:
            self.make_plot('valid', batch_idx, glyphs, images)
        return lss

    def test_step(self, test_batch, batch_idx):
        glyphs, vectors = test_batch
        images = self.forward(vectors).reshape(-1, 1, 96, 96)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        lss = self.benchmark(images, glyphs)
        self.log('test_loss', lss)
        self.print_errors(images, glyphs)
        self.make_plot('test', batch_idx, glyphs, images)
        return lss

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(ds.ParquetDataset("data/dataset/train.parquet"), batch_size=10, num_workers=8, shuffle=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(ds.ParquetDataset("data/dataset/test.parquet"), batch_size=10, num_workers=8, shuffle=False)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(ds.ParquetDataset("data/dataset/validation.parquet"), batch_size=10, num_workers=8, shuffle=False)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass


class Baseline(AbstractNet):
    def __init__(self):
        super().__init__()
        self.model_name = 'bs'
        self.benchmark = nn.MSELoss()
        self.transformer = nn.Transformer(d_model=4, nhead=2)

    def forward(self, vector):
        b = vector.shape[0]
        vector = vector.reshape(b, 50, 2)
        src = th.cat((vector, th.zeros_like(vector)), dim=2).reshape(b, 50, 4)
        tgt = th.zeros_like(src)
        out = self.transformer(src, tgt)
        curve = out[:, :, 0:2].reshape(b, 50, 2)
        widthx = out[:, :, 2:3].reshape(b, 50, 1, 1)
        widthy = out[:, :, 3:4].reshape(b, 50, 1, 1)
        return stroke(curve, widthx, widthy, 1.0)

    def loss(self, predict, target):
        return self.benchmark(predict, target)


_model_ = Baseline

