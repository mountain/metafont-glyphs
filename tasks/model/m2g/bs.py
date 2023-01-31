import torch as th
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import data.dataset as ds
import util.basemap as bm

from abc import ABC
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader


if th.cuda.is_available():
    pass


class AbstractNet(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.lr = 0.02

    def benchmark(self, predict, targets):
        pprod, pylds = predict
        tprod, tylds = targets

        pttl = th.sum(pprod * country_area_weights, dim=(2, 3, 4))
        perr = (pttl - tprod) * (pttl - tprod)

        yyld = pylds * country_area_weights
        ymns = th.sum(yyld, dim=(2, 3, 4)) / (1e-7 + th.sum(country_area_weights * (yyld > 0), dim=(2, 3, 4)))
        yerr = (ymns - tylds) * (ymns - tylds)

        return th.mean(th.sum(perr, dim=1)) + th.mean(th.mean(yerr, dim=1))

    def print_errors(self, predict, targets):
        print('----------------------------------------------------------------')
        for ix, key in enumerate(ds.countries):
            print(key, 'product', tprod[0, ix].item(), prmse[0, ix].item(), pmax[0, ix].item(), 'yeilds', tylds[0, ix].item(), yrmse[0, ix].item(), ymax[0, ix].item())
        print('----------------------------------------------------------------')

    def make_plot(self, xs, ys):
        years, data = xs
        prods, ylds = ys
        for yr, pd, yd in zip(years, prods, ylds):
            pd = pd[0, 0, :, :]
            yd = yd[0, 0, :, :]
            pd = pd.cpu().numpy().reshape([73, 144])
            yd = yd.cpu().numpy().reshape([73, 144])
            pd = np.roll(pd, 72, axis=1)
            yd = np.roll(yd, 72, axis=1)

            bm.plot('outputs/%s-prod-%d' % (self.model_name, yr), pd)
            bm.plot('outputs/%s-ylds-%d' % (self.model_name, yr), yd)

    def forward(self, years, data):
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
            self.make_plot(xs, ys)
        return lss

    def test_step(self, test_batch, batch_idx):
        xs, yt = test_batch
        ys = self.forward(*xs)
        lss = self.benchmark(ys, yt)
        self.log('test_loss', lss)
        self.print_errors(ys, yt)
        self.make_plot(xs, ys)
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
        self.param = nn.Parameter(th.ones(1, 1, 1, 1, 1))
        self.model_name = 'bs'

    def forward(self, years, data):
        pttal = pcoeff * years + pbias
        prod = (pttal / land_total).reshape([-1, 1, 1, 1, 1])
        pred = prod * land.reshape([-1, 1, 1, 73, 144]) * self.param

        yttal = ycoeff * years + ybias
        ylds = (yttal / land_total).reshape([-1, 1, 1, 1, 1])
        ylds = ylds * land.reshape([-1, 1, 1, 73, 144]) * self.param

        return pred, ylds

    def loss(self, predict, target):
        return self.benchmark(predict, target)


_model_ = Baseline

