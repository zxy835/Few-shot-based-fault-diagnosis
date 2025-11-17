import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.relu = nn.ReLU()
        self.fuse = nn.Conv1d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        out = torch.cat([out3, out5, out7], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.fuse(out)
        return out


class ResidualMultiScaleCNN(nn.Module):
    def __init__(self, in_channels, base_channels=64, embed_dim=512, num_blocks=4,
                 return_residuals=False, use_nonlinear_residual=False):
        super().__init__()
        self.return_residuals = return_residuals
        self.use_nonlinear_residual = use_nonlinear_residual

        self.initial_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU()

        self.blocks = nn.ModuleList([
            MultiScaleConvBlock(base_channels, base_channels)
            for _ in range(num_blocks)
        ])

        if return_residuals and use_nonlinear_residual:
            self.res_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(base_channels, base_channels, kernel_size=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(base_channels)
                ) for _ in range(num_blocks)
            ])

        self.pool = nn.AdaptiveAvgPool1d(6)
        self.fc = nn.Linear(base_channels * 6, embed_dim)

    def forward(self, x):
        out = self.initial_conv(x)
        out = self.bn(out)
        out = self.relu(out)

        residual_outputs = []

        for i, block in enumerate(self.blocks):
            res = out
            out = block(out)
            if self.return_residuals:
                if self.use_nonlinear_residual:
                    res_out = self.res_adapters[i](res)
                else:
                    res_out = res
                residual_outputs.append(res_out)
            out = out + res
            out = self.relu(out)

        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        if self.return_residuals:
            return out, residual_outputs
        return out


class TeacherStudentMSCNN(nn.Module):
    def __init__(self, in_channels=6, base_channels=64, embed_dim=512):
        super().__init__()
        # 教师模型：输入为 early + severe，结构简单，不做非线性残差
        self.teacher = ResidualMultiScaleCNN(
            in_channels * 2, base_channels, embed_dim,
            num_blocks=2, return_residuals=True, use_nonlinear_residual=False)

        # 学生模型：输入为 early，结构深，并增强残差路径
        self.student = ResidualMultiScaleCNN(
            in_channels, base_channels, embed_dim,
            num_blocks=2, return_residuals=True, use_nonlinear_residual=True)

    def forward_teacher(self, x_early, x_severe):
        combined = torch.cat([x_early, x_severe], dim=1)
        embed, residuals = self.teacher(combined)
        return embed, residuals

    def forward_student(self, x_early):
        embed, residuals = self.student(x_early)
        return embed, residuals
