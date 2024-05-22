import torch as th
import torch.nn as nn
import lightning as ltn
import data.dataset as ds

from torchvision.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader


class AbstractG2MNet(ltn.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 0.0001
        self.benchmark = nn.MSELoss()

    def forward(self, vector):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss(self, predict, target):
        raise NotImplementedError()

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
        strokes = self.forward(glyphs).reshape(-1, 1, 96, 96) * 0.98 + 0.01
        lss = self.loss(strokes, vectors)
        self.log('val_loss', lss, prog_bar=True)
        return lss

    def test_step(self, test_batch, batch_idx):
        glyphs, vectors = test_batch
        vectors = vectors.reshape(-1, 100)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        strokes = self.forward(glyphs).reshape(-1, 1, 96, 96) * 0.98 + 0.01
        lss = self.loss(strokes, vectors)
        self.log('test_loss', lss)
        return lss

    def train_dataloader(self):
        return DataLoader(ds.ParquetDataset("../../data/dataset/train.parquet"), batch_size=20, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(ds.ParquetDataset("../../data/dataset/validation.parquet"), batch_size=10, num_workers=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(ds.ParquetDataset("../../data/dataset/test.parquet"), batch_size=10, num_workers=1, shuffle=False)


class Baseline(AbstractG2MNet):
    def __init__(self):
        super().__init__()
        self.model_name = 'bs'
        self.vit = VisionTransformer(
            image_size=96, patch_size=16, num_layers=1, num_heads=8, num_classes=100,
            hidden_dim=512, mlp_dim=128, dropout=0.1, attention_dropout=0.1
        )

    def forward(self, glyph):
        return self.vit(glyph)


_model_ = Baseline
