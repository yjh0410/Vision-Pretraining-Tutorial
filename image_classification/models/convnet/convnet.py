import torch
import torch.nn as nn

from .modules import ConvModule


# Convolutional Network
class ConvNet(nn.Module):
    def __init__(self,
                 in_dim     :int,
                 inter_dim  :int,
                 out_dim    :int,
                 act_type   :str = "sigmoid",
                 norm_type  :str = "bn",
                 avgpool    :bool = True,
                 ) -> None:
        super().__init__()
        self.stem   = ConvModule(in_dim, inter_dim, kernel_size=3, padding=1, stride=1, act_type=act_type, norm_type=norm_type)
        self.layers = nn.Sequential(
            ConvModule(inter_dim, inter_dim, kernel_size=3, padding=1, stride=1, act_type=act_type, norm_type=norm_type),
            nn.MaxPool2d([2, 2], stride=2),
            ConvModule(inter_dim, inter_dim, kernel_size=3, padding=1, stride=1, act_type=act_type, norm_type=norm_type),
            nn.MaxPool2d([2, 2], stride=2),
            ConvModule(inter_dim, inter_dim, kernel_size=3, padding=1, stride=1, act_type=act_type, norm_type=norm_type),
            nn.MaxPool2d([2, 2], stride=2),
            ConvModule(inter_dim, inter_dim, kernel_size=3, padding=1, stride=1, act_type=act_type, norm_type=norm_type),
            )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if avgpool else None
        self.fc     = nn.Linear(inter_dim, out_dim)

    def forward(self, x):
        """
        Input:
            x : (torch.Tensor) -> [B, C, H, W]
        """
        x = self.stem(x)
        x = self.layers(x)

        if self.avgpool is not None:
            x = self.avgpool(x)

        # [B, C1, H, W] -> [B, C2], C2 = C1 x H x W
        x = x.flatten(1)
        x = self.fc(x)

        return x
