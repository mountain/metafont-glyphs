import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as ltn
import data.dataset as ds

from torch.utils.data import DataLoader
from util.stroke import IX, IY

dltrain = DataLoader(
    ds.VocabDataset("../../data/dataset/train.parquet"),
    batch_size=256, num_workers=16, shuffle=True, persistent_workers=True
)
dlvalid = DataLoader(
    ds.VocabDataset("../../data/dataset/validation.parquet"),
    batch_size=32, num_workers=8, shuffle=False, persistent_workers=True
)
dltest = DataLoader(
    ds.VocabDataset("../../data/dataset/test.parquet"),
    batch_size=32, num_workers=8, shuffle=False, persistent_workers=True
)


class ImageVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super(ImageVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: 编码图像
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (batch_size, 32, 48, 48)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (batch_size, 64, 24, 24)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (batch_size, 128, 12, 12)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (batch_size, 256, 6, 6)
            nn.ReLU(),
            nn.Flatten(),
        )

        # 潜在空间
        self.fc_mu = nn.Linear(256 * 6 * 6, latent_dim)
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_dim)

        # 解码器：从潜在空间解码
        self.decoder_input = nn.Linear(latent_dim, 256 * 6 * 6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (batch_size, 128, 12, 12)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (batch_size, 64, 24, 24)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (batch_size, 32, 48, 48)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # (batch_size, 1, 96, 96)
            nn.Sigmoid()
        )

    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        decoded_input = self.decoder_input(z).view(-1, 256, 6, 6)
        return self.decoder(decoded_input)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def compute_loss(self, reconstructed, original, mu, logvar):
        reconstruction_loss = F.mse_loss(reconstructed, original, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence


class VocabVAE(nn.Module):
    def __init__(self, vocab_size, latent_dim=64, max_seq_length=80, embedding_dim=128):
        super(VocabVAE, self).__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_rnn = nn.GRU(embedding_dim, 256, batch_first=True)

        # 潜在空间
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # 解码器
        self.decoder_input = nn.Linear(latent_dim, 256)
        self.decoder_rnn = nn.GRU(embedding_dim, 256, batch_first=True)
        self.fc_out = nn.Linear(256, vocab_size)

        self.max_seq_length = max_seq_length

    def encode(self, x):
        embedded_input = self.embedding(x)
        _, hidden = self.encoder_rnn(embedded_input)
        hidden = hidden.squeeze(0)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        decoder_hidden = self.decoder_input(z).unsqueeze(0)
        decoder_input = torch.zeros(z.size(0), seq_len, dtype=torch.long).to(z.device)
        outputs = []

        # 逐步解码
        for t in range(seq_len):
            embed = self.embedding(decoder_input[:, t].unsqueeze(1))
            output, decoder_hidden = self.decoder_rnn(embed, decoder_hidden)
            output = self.fc_out(output.squeeze(1))
            decoder_input = decoder_input.clone()  # 避免 in-place 操作，使用克隆
            decoder_input[:, t] = torch.argmax(output, dim=-1)  # 非 in-place 操作
            outputs.append(output)

        return torch.stack(outputs, dim=1)

    def forward(self, x):
        seq_len = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded_output = self.decode(z, seq_len)
        return decoded_output, mu, logvar

    def compute_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.cross_entropy(recon_x.view(-1, recon_x.size(-1)), x.view(-1), reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_divergence


class DualVAE(ltn.LightningModule):
    def __init__(self, vocab_size=6149, latent_dim=64):
        super(DualVAE, self).__init__()
        self.image_vae = ImageVAE(latent_dim=latent_dim)
        self.vocab_vae = VocabVAE(vocab_size=vocab_size, latent_dim=latent_dim)
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=ds.VOCAB2ID[ds.STARTER])

    def forward(self, glyphs, code_seq=None):
        # 图像部分 VAE 的前向传播
        glyphs = glyphs.reshape(-1, 1, 96, 96)  # 确保 glyphs 是 (batch_size, 1, 96, 96)
        img_reconstructed, mu_img, logvar_img = self.image_vae(glyphs)  # 返回重构图像、均值和方差
        latent_img = self.image_vae.reparameterize(mu_img, logvar_img)  # 获取潜在变量 latent_img

        if code_seq is not None:  # 训练阶段
            # 代码部分 VAE 的前向传播
            code_reconstructed, mu_code, logvar_code = self.vocab_vae(code_seq)  # 返回重构代码、均值和方差
            latent_code = self.vocab_vae.reparameterize(mu_code, logvar_code)  # 获取潜在变量 latent_code
            return latent_img, latent_code, (img_reconstructed, mu_img, logvar_img), (
            code_reconstructed, mu_code, logvar_code)
        else:  # 推理阶段，使用图像的潜在向量生成代码
            recon_code, mu_code, logvar_code = self.vocab_vae.decode(latent_img, seq_len=80)
            return latent_img, (recon_code, mu_code, logvar_code)

    def compute_loss(self, logits, labels, img_result, code_result, glyphs):
        # 分类损失 (只应用于代码重构的结果)
        classification_loss = self.cross_entropy_loss(logits.view(-1, logits.size(-1)), labels.view(-1))

        # 图像部分的 VAE 损失 (mse_loss 应用于图像的重构)
        img_reconstructed, mu_img, logvar_img = img_result
        glyphs = glyphs.reshape(-1, 1, 96, 96)  # 确保 glyphs 形状正确
        img_vae_loss = self.image_vae.compute_loss(img_reconstructed, glyphs, mu_img,
                                                   logvar_img)  # 使用原始图像 glyphs 作为原始图像数据

        # 代码部分的 VAE 损失 (cross_entropy_loss 应用于代码序列的重构)
        code_reconstructed, mu_code, logvar_code = code_result
        code_vae_loss = self.vocab_vae.compute_loss(code_reconstructed, labels, mu_code, logvar_code)

        return classification_loss + img_vae_loss + code_vae_loss

    def training_step(self, batch, batch_idx):
        glyphs, labels = batch
        labels = labels.reshape(-1, 80)
        # 前向传播并返回潜在向量和结果
        latent_img, latent_code, img_result, code_result = self(glyphs, labels)
        # 计算损失
        loss = self.compute_loss(code_result[0], labels, img_result, code_result, glyphs)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        glyphs, labels = batch
        labels = labels.reshape(-1, 80)
        # 前向传播并返回潜在向量和结果
        latent_img, latent_code, img_result, code_result = self(glyphs, labels)
        # 计算损失
        loss = self.compute_loss(code_result[0], labels, img_result, code_result, glyphs)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def train_dataloader(self):
        return dltrain

    def val_dataloader(self):
        return dlvalid

    def test_dataloader(self):
        return dltest


_model_ = DualVAE
