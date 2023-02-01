import os

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
        self.lr = 0.0001
        self.benchmark = nn.MSELoss()

    def make_plot(self, iname, ix, xs, ys):
        xd = xs[0, 0, :, :]
        yd = ys[0, 0, :, :]
        src = np.array(xd.cpu().numpy().reshape((96, 96)) * 255, dtype=np.uint8)
        tgt = np.array(yd.cpu().numpy().reshape((96, 96)) * 255, dtype=np.uint8)
        os.makedirs('temp/outputs', exist_ok=True)
        cv2.imwrite('temp/outputs/%s-%03d-%03d.png' % ('o', self.current_epoch, ix), src)
        cv2.imwrite('temp/outputs/%s-%03d-%03d.png' % (iname, self.current_epoch, ix), tgt)

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
        lss = self.loss(images, glyphs)
        self.log('train_loss', lss)
        return lss

    def validation_step(self, val_batch, batch_idx):
        glyphs, vectors = val_batch
        images = self.forward(vectors).reshape(-1, 1, 96, 96) * 0.98 + 0.01
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        lss = self.loss(images, glyphs)
        self.log('val_loss', lss, prog_bar=True)
        if batch_idx % 10 == 9:
            self.make_plot('v', batch_idx, glyphs, images)
        return lss

    def test_step(self, test_batch, batch_idx):
        glyphs, vectors = test_batch
        images = self.forward(vectors).reshape(-1, 1, 96, 96)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        lss = self.loss(images, glyphs)
        self.log('test_loss', lss)
        self.print_errors(images, glyphs)
        self.make_plot('t', batch_idx, glyphs, images)
        return lss

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(ds.ParquetDataset("data/dataset/train.parquet"), batch_size=20, num_workers=8, shuffle=True)

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
        self.transformer = nn.Transformer(d_model=5, nhead=5)
        self.lc = nn.Linear(5, 5, bias=True)

    def forward(self, vector):
        b = vector.shape[0]
        s = ds.maxlen // 2
        vector = vector.reshape(b, s, 2)
        src = th.cat((vector, th.zeros_like(vector), th.zeros_like(vector[:, :, 0:1])), dim=2).reshape(b, s, 5)
        tgt = th.zeros_like(src)
        out = th.sigmoid(self.lc(self.transformer(src, tgt)).reshape(b, s, 5))
        curve = out[:, :, 0:2].reshape(b, s, 2)
        widthx = out[:, :, 2:3].reshape(b, s, 1, 1)
        widthy = out[:, :, 3:4].reshape(b, s, 1, 1)
        densty = out[:, :, 3:4].reshape(b, s, 1, 1) * 10
        return stroke(curve, widthx, widthy, densty)

    def loss(self, predict, target):
        return self.benchmark(predict, target)


_model_ = Baseline

