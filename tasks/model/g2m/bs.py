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
        self.lr = 0.001
        self.mse = nn.MSELoss()

    def forward(self, vector):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, 53)
        return [optimizer], [scheduler]

    def loss(self, predict, target):
        return self.mse(predict, target)

    def training_step(self, train_batch, batch_idx):
        glyphs, vectors = train_batch
        vectors = vectors.reshape(-1, 80)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        strokes = self.forward(glyphs).reshape(-1, 80)
        lss = self.loss(strokes, vectors)
        self.log('train_loss', lss, prog_bar=True)
        return lss

    def validation_step(self, val_batch, batch_idx):
        glyphs, vectors = val_batch
        vectors = vectors.reshape(-1, 80)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        strokes = self.forward(glyphs).reshape(-1, 80)
        lss = self.loss(strokes, vectors)
        self.log('val_loss', lss, prog_bar=True)
        return lss

    def test_step(self, test_batch, batch_idx):
        glyphs, vectors = test_batch
        vectors = vectors.reshape(-1, 80)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        strokes = self.forward(glyphs).reshape(-1, 80)
        lss = self.loss(strokes, vectors)
        self.log('test_loss', lss)
        return lss

    def train_dataloader(self):
        return DataLoader(ds.ParquetDataset("../../data/dataset/train.parquet"), batch_size=32, num_workers=2, shuffle=True, drop_last=True, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(ds.ParquetDataset("../../data/dataset/validation.parquet"), batch_size=32, num_workers=2, shuffle=False, drop_last=True, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(ds.ParquetDataset("../../data/dataset/test.parquet"), batch_size=32, num_workers=2, shuffle=False, drop_last=True, pin_memory=True, prefetch_factor=2, persistent_workers=True)

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
        self.vit = VisionTransformer(
            image_size=96, patch_size=16, num_layers=6, num_heads=16, num_classes=80,
            hidden_dim=128, mlp_dim=256, dropout=0.1, attention_dropout=0.1
        )
        for ix in range(6):
            self.vit.encoder.layers[ix].mlp[1] = OptAEGV3()

    def forward(self, glyph):
        xslice = IX.to(glyph.device) * th.ones_like(glyph)
        yslice = IY.to(glyph.device) * th.ones_like(glyph)
        data = th.cat([glyph, xslice, yslice], dim=1)
        return self.vit(data)


_model_ = Baseline

