import torch
import torch.nn as nn

class CrossAttentionGateFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4):
        super().__init__()

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=1024,
                batch_first=True
            ),
            num_layers=2
        )
        self.attn_weights = None
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x1, x2, x3, x4, x5, return_attn=False):
        """
        输入：
            x1: (batch, embed_dim) 需要诊断的故障信号embedding（query）
            x2~x5: (batch, embed_dim) 四个基准故障embedding（support）
        输出：
            distances: (batch, 4) x1与x2~x5的欧式距离（相似度）
        """

        batch_size = x1.size(0)

        # 1. 扩展序列维度(模拟seq_len=1)
        q = x1.unsqueeze(1)  # (batch, 1, embed_dim)
        s = torch.stack([x2, x3, x4, x5], dim=1)  # (batch, 4, embed_dim)

        # 2. Transformer编码
        q_enc = self.transformer_encoder(q)  # (batch, 1, embed_dim)
        s_enc = self.transformer_encoder(s)  # (batch, 4, embed_dim)

        # 3. 跨注意力：query作为query，support作为key和value
        attn_out, attn_weights = self.cross_attn(
            query=q_enc,
            key=s_enc,
            value=s_enc,
            need_weights=True,  # 必须加，否则不返回注意力
            average_attn_weights=False  # 保留每个 head
        )
        # attn_weights: (B, num_heads, 1, 4)  query=1位置，4个support位置
        attn_weights = attn_weights.detach().clone()
        self.attn_weights = attn_weights

        # 4. 残差连接 + LayerNorm
        q_attn = self.ln1(q_enc + attn_out)  # (batch, 1, embed_dim)

        # 5. 门控融合
        # 扩展q_attn以匹配s_enc维度
        q_exp = q_attn.expand(-1, 4, -1)  # (batch, 4, embed_dim)

        combined = torch.cat([q_exp, s_enc], dim=-1)  # (batch, 4, embed_dim*2)
        gate = self.gate(combined)  # (batch, 4, 1)

        fused = gate * q_exp + (1 - gate) * s_enc  # (batch, 4, embed_dim)
        fused = self.ln2(fused)

        # 6. 计算q_attn与fused的欧式距离
        q_vec = q_attn.squeeze(1)  # (batch, embed_dim)

        distances = torch.norm(q_vec.unsqueeze(1) - fused, p=2, dim=-1)  # (batch, 4)

        if return_attn:
            # 压缩维度 (B, H, 4) → 可以后续平均
            attn = self.attn_weights.squeeze(2)
            return distances, attn  # ✅ 返回 model output & attention map

        return distances
