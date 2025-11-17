import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.state_proj = nn.Linear(d_model, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        x: [B, L, C]  (sequence-first, like in transformers)
        """
        residual = x
        x = self.norm(x)
        gate = torch.sigmoid(self.gate(x))               # 类似门控机制
        state = self.activation(self.state_proj(x))      # 状态建模
        out = self.output_proj(state)
        return residual + gate * out                     # 门控残差连接
