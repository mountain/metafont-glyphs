import torch as th

from torch import nn
from torchvision.models import VisionTransformer

from g2m.bs import AbstractG2MNet
from util.stroke import IX, IY


def batch_aeg_product_optimized(A, B):
    """
    优化后的 batch_aeg_product 函数
    A: (batch_size, rows, features)
    B: (batch_size, features, cols)
    返回: (batch_size, rows, cols) 的结果
    """
    N, rows, features = A.shape
    _, _, cols = B.shape

    # 初始化结果张量
    result = th.zeros(N, rows, cols, device=A.device, dtype=A.dtype)

    # 创建行和列的索引
    i_indices = th.arange(rows, device=A.device).view(rows, 1).expand(rows, cols)  # (rows, cols)
    j_indices = th.arange(cols, device=A.device).view(1, cols).expand(rows, cols)  # (rows, cols)

    for k in range(features):
        # 计算 mask，其中 mask[i,j] = ((i + j + k) % 2 == 0)
        mask = ((i_indices + j_indices + k) % 2 == 0).to(A.device)  # (rows, cols)

        # 获取 A 和 B 中第 k 个特征
        A_k = A[:, :, k].unsqueeze(2)  # (N, rows, 1)
        B_k = B[:, k, :].unsqueeze(1)  # (N, 1, cols)

        # 根据 mask 更新结果
        # 使用广播机制使 mask 适应 (N, rows, cols)
        mask_broadcast = mask.unsqueeze(0)  # (1, rows, cols)
        mask_broadcast = mask_broadcast.expand(N, rows, cols)  # (N, rows, cols)

        # 计算 (result + x) * y 和 (result + y) * x
        option1 = (result + A_k) * B_k  # 当 mask 为 True 时使用
        option2 = (result + B_k) * A_k  # 当 mask 为 False 时使用

        # 使用 torch.where 根据 mask 选择对应的选项
        result = th.where(mask_broadcast, option1, option2)

    return result


class SemiLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SemiLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(th.Tensor(1, out_features, in_features))
        self.proj = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)

    def forward(self, input):
        expanded_weight = self.weight.expand(input.size(0), -1, -1)  # (batch_size, out_features, in_features)
        reshaped_input = input.view(input.size(0), input.size(1), 1)  # (batch_size, in_features, 1)
        aeg_result = batch_aeg_product_optimized(expanded_weight, reshaped_input)  # (batch_size, out_features, 1)
        aeg_result = aeg_result.squeeze(2)  # (batch_size, out_features)
        return th.sigmoid(aeg_result) * self.proj(input)


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


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(ViTBlock, self).__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.msa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            OptAEGV3(),
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

        # Regression head
        self.regression_head = nn.Sequential(
            OptAEGV3(),
            nn.Linear(embed_dim, num_outputs)
        )

    def forward(self, x):
        # Divide image into patches
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height // self.patch_size, self.patch_size, width // self.patch_size,
                   self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(batch_size, -1, self.patch_size * self.patch_size * channels)

        # Patch embedding
        x = self.patch_embed(x)

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

        # Pass through the regression head
        output = self.regression_head(x)
        return output


class AEGModel(AbstractG2MNet):
    def __init__(self):
        super().__init__()
        self.model_name = 'aeg'
        self.vit16 = ViT(
            image_size=96, patch_size=16, num_layers=4, num_heads=16, mlp_dim=512, num_outputs=80
        )

    def forward(self, glyph):
        xslice = IX.to(glyph.device) * th.ones_like(glyph)
        yslice = IY.to(glyph.device) * th.ones_like(glyph)
        data = th.cat([glyph, xslice, yslice], dim=1)
        return self.vit16(data)


_model_ = AEGModel

