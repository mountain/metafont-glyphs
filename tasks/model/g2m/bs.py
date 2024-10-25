import torch as th
import torch.nn as nn
import lightning as ltn
import data.dataset as ds

from torch.utils.data import DataLoader
from util.stroke import IX, IY


dltrain = DataLoader(
    ds.VocabDataset("../../data/dataset/train.parquet"),
    batch_size=32, num_workers=2, shuffle=True, persistent_workers=True
)
dlvalid = DataLoader(
    ds.VocabDataset("../../data/dataset/validation.parquet"),
    batch_size=32, num_workers=2, shuffle=False, persistent_workers=True
)
dltest = DataLoader(
    ds.VocabDataset("../../data/dataset/test.parquet"),
    batch_size=32, num_workers=2, shuffle=False, persistent_workers=True
)


class AbstractG2MNet(ltn.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 0.001
        self.celoss = nn.CrossEntropyLoss(ignore_index=ds.VOCAB2ID[ds.STARTER])

    def forward(self, vector, labels=None):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, 53)
        return [optimizer], [scheduler]

    def loss(self, logits, labels):
        loss = self.celoss(logits, labels.view(-1))
        return loss

    def training_step(self, train_batch, batch_idx):
        glyphs, labels = train_batch
        labels = labels.reshape(-1, 80)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        strokes = self.forward(glyphs, labels=labels).reshape(-1, 80)  # 传入 labels
        lss = self.loss(strokes, labels)
        self.log('train_loss', lss, prog_bar=True)
        return lss

    def validation_step(self, val_batch, batch_idx):
        glyphs, labels = val_batch
        labels = labels.reshape(-1, 80)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        strokes = self.forward(glyphs).reshape(-1, 80)  # 传入 labels
        lss = self.loss(strokes, labels)
        self.log('val_loss', lss, prog_bar=True)
        return lss

    def test_step(self, test_batch, batch_idx):
        glyphs, labels = test_batch
        labels = labels.reshape(-1, 80)
        glyphs = glyphs.reshape(-1, 1, 96, 96)
        strokes = self.forward(glyphs).reshape(-1, 80)
        lss = self.loss(strokes, labels)
        self.log('test_loss', lss)
        return lss

    def train_dataloader(self):
        return dltrain

    def val_dataloader(self):
        return dlvalid

    def test_dataloader(self):
        return dltest

    def on_save_checkpoint(self, checkpoint):
        print()


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(ViTBlock, self).__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.msa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-head self-attention (MSA)
        x = x + self.msa(self.layernorm1(x), self.layernorm1(x), self.layernorm1(x))[0]
        # Feed-forward network (MLP)
        x = x + self.mlp(self.layernorm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_layers=12, num_heads=12, embed_dim=768, mlp_dim=3072,
                 num_outputs=10):
        super(ViT, self).__init__()

        # Patch embedding
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, embed_dim)

        # Class token and position embeddings
        self.cls_token = nn.Parameter(th.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(th.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)
        ])

        # Layer normalization before the regression head
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Divide image into patches
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height // self.patch_size, self.patch_size, width // self.patch_size,
                   self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(batch_size, -1, self.patch_size * self.patch_size * channels)
        # print("rearrange:", x.shape)

        # Patch embedding
        x = self.patch_embed(x)
        # print("embedding:", x.shape)

        # Add class token and position embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = th.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Take the class token output
        x = self.layernorm(x[:, 0])

        return x


class ConditionalTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_decoder_layers, dim_feedforward, max_seq_length):
        super(ConditionalTransformerDecoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(th.zeros(1, max_seq_length, d_model))
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)  # 输出层

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        tgt_emb = self.embedding(tgt) * th.sqrt(th.tensor(self.d_model, dtype=th.float32))
        tgt_emb = tgt_emb + self.positional_encoding[:, :tgt_emb.size(1), :]
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.fc_out(output)
        return output


class Baseline(AbstractG2MNet):
    def __init__(self):
        super().__init__()
        self.model_name = 'bs'
        self.encoder = ViT(
            image_size=96, patch_size=16, num_layers=6, num_heads=16, embed_dim=128, mlp_dim=256
        )
        self.decoder = ConditionalTransformerDecoder(
            vocab_size=len(ds.VOCAB2ID),
            d_model=128, num_heads=16, num_decoder_layers=6, dim_feedforward=256, max_seq_length=80
        )

    def forward(self, glyph, labels=None):
        xslice = IX.to(glyph.device) * th.ones_like(glyph)
        yslice = IY.to(glyph.device) * th.ones_like(glyph)
        data = th.cat([glyph, xslice, yslice], dim=1)

        conditional = self.encoder(data)
        conditional = conditional.unsqueeze(1)

        outputs = [] # 用于存储每个时间步的输出 logits
        tgt = th.zeros(glyph.size(0), 1, dtype=th.long).to(glyph.device)
        tgt[:, 0] = ds.VOCAB2ID[ds.STARTER]

        # 如果是训练阶段，使用真实的 labels
        if labels is not None:
            for i in range(79):
                output = self.decoder(tgt, conditional) # 获取当前时间步的输出 logits
                outputs.append(output)
                tgt = labels[:, i].unsqueeze(1)  # 使用真实的笔画作为输入
        # 如果是推理阶段，使用预测的笔画
        else:
            for i in range(79):
                output = self.decoder(tgt, conditional) # 获取当前时间步的输出 logits
                outputs.append(output)
                tgt = th.argmax(output, dim=-1)  # 使用预测的笔画作为输入

        outputs = th.cat(outputs, dim=1) # 将所有时间步的输出 logits 连接起来
        return outputs


_model_ = Baseline

