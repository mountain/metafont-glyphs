import torch as th
import torch.nn as nn
import lightning as ltn
import data.dataset as ds

from torchvision.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader
from util.stroke import IX, IY


class AbstractG2MNet(ltn.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 0.0001
        self.mse = nn.MSELoss()

    def forward(self, vector):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss(self, predict, target):
        return self.mse(predict, target)

    def training_step(self, train_batch, batch_idx):
        glyphs, vectors = train_batch
        vectors = vectors.reshape(-1, 100)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        strokes = self.forward(glyphs).reshape(-1, 100)
        lss = self.loss(strokes, vectors)
        self.log('train_loss', lss, prog_bar=True)
        return lss

    def validation_step(self, val_batch, batch_idx):
        glyphs, vectors = val_batch
        vectors = vectors.reshape(-1, 100)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        strokes = self.forward(glyphs).reshape(-1, 100)
        lss = self.loss(strokes, vectors)
        self.log('val_loss', lss, prog_bar=True)
        return lss

    def test_step(self, test_batch, batch_idx):
        glyphs, vectors = test_batch
        vectors = vectors.reshape(-1, 100)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        strokes = self.forward(glyphs).reshape(-1, 100)
        lss = self.loss(strokes, vectors)
        self.log('test_loss', lss)
        return lss

    def train_dataloader(self):
        return DataLoader(ds.ParquetDataset("../../data/dataset/train.parquet"), batch_size=128, num_workers=8, shuffle=True, drop_last=True, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(ds.ParquetDataset("../../data/dataset/validation.parquet"), batch_size=64, num_workers=3, shuffle=False, drop_last=True, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(ds.ParquetDataset("../../data/dataset/test.parquet"), batch_size=64, num_workers=3, shuffle=False, drop_last=True, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    def on_save_checkpoint(self, checkpoint):
        print()


class OptAEGV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.vx = nn.Parameter(th.zeros(1, 1, 1))
        self.vy = nn.Parameter(th.ones(1, 1, 1))
        self.wx = nn.Parameter(th.zeros(1, 1, 1))
        self.wy = nn.Parameter(th.ones(1, 1, 1))
        self.afactor = nn.Parameter(th.zeros(1, 1))
        self.mfactor = nn.Parameter(th.ones(1, 1))

    def flow(self, dx, dy, data):
        return data * (1 + dy) + dx

    def forward(self, data):
        shape = data.size()
        data = data.flatten(1)
        data = data - data.mean()
        data = data / data.std()

        b = shape[0]
        v = self.flow(self.vx, self.vy, data.view(b, -1, 1))
        w = self.flow(self.wx, self.wy, data.view(b, -1, 1))

        dx = self.afactor * th.sum(v * th.sigmoid(w), dim=-1)
        dy = self.mfactor * th.tanh(data)
        data = self.flow(dx, dy, data)

        return data.view(*shape)


class Baseline(AbstractG2MNet):
    def __init__(self):
        super().__init__()
        self.model_name = 'bs'
        self.dnsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv0 = nn.Conv2d(3, 9, kernel_size=5, padding=2)
        self.nlon0 = OptAEGV3()
        self.conv1 = nn.Conv2d(9, 27, kernel_size=3, padding=1)
        self.nlon1 = OptAEGV3()
        self.conv2 = nn.Conv2d(27, 81, kernel_size=3, padding=1)
        self.nlon2 = OptAEGV3()
        self.conv3 = nn.Conv2d(27 + 81, 36, kernel_size=3, padding=1)
        self.nlon3 = OptAEGV3()
        self.conv4 = nn.Conv2d(9 + 36, 15, kernel_size=3, padding=1)
        self.nlon4 = OptAEGV3()
        self.conv5 = nn.Conv2d(3 + 15, 6, kernel_size=3, padding=1)
        self.nlon5 = OptAEGV3()
        self.vit = VisionTransformer(
            image_size=96, patch_size=16, num_layers=6, num_heads=16, num_classes=100,
            hidden_dim=512, mlp_dim=256, dropout=0.1, attention_dropout=0.1
        )
        first_conv = self.vit.conv_proj
        self.vit.conv_proj = nn.Conv2d(6, first_conv.out_channels, kernel_size=first_conv.kernel_size, stride=first_conv.stride, padding=first_conv.padding)
        for ix in range(6):
            self.vit.encoder.layers[ix].mlp[1] = OptAEGV3()

    def forward(self, glyph):
        xslice = IX.to(glyph.device) * th.ones_like(glyph)
        yslice = IY.to(glyph.device) * th.ones_like(glyph)
        data = th.cat([glyph, xslice, yslice], dim=1)
        data0 = self.conv0(data)
        data0 = self.nlon0(data0)
        data1 = self.dnsample(data0)
        data1 = self.conv1(data1)
        data1 = self.nlon1(data1)
        data2 = self.dnsample(data1)
        data2 = self.conv2(data2)
        data2 = self.nlon2(data2)
        data3 = self.upsample(data2)
        data3 = th.cat([data3, data1], dim=1)
        data3 = self.conv3(data3)
        data3 = self.nlon3(data3)
        data4 = self.upsample(data3)
        data4 = th.cat([data4, data0], dim=1)
        data4 = self.conv4(data4)
        data4 = self.nlon4(data4)
        data5 = self.upsample(data4)
        data5 = th.cat([data5, data], dim=1)
        data5 = self.conv5(data5)
        data5 = self.nlon5(data5)
        return self.vit(data5)


_model_ = Baseline

