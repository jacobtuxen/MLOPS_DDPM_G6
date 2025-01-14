from types import SimpleNamespace

import torch


class UNet(torch.nn.Module):
    def __init__(self, depth: int, in_channels: int, out_channels: int, size: int) -> None:
        super().__init__()
        in_channels += 1

        self.depth = depth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.size = size
        self.maxpool = torch.nn.MaxPool2d(2)
        self.config = SimpleNamespace()
        self.config.in_channels = in_channels
        self.config.sample_size = size

        self.down = torch.nn.ModuleList()
        for i in range(depth):
            self.down.append(self.unet_down_block(in_channels, 2**i * 16))
            in_channels = 2**i * 16

        self.bottle_neck = self.double_conv(in_channels, 2 * in_channels)

        self.up = torch.nn.ModuleList()
        self.up_conv = torch.nn.ModuleList()
        for i in range(depth):
            self.up_conv.append(self.double_conv(in_channels * 2, in_channels))
            self.up.append(torch.nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=2, stride=2))
            in_channels = in_channels // 2

        self.out = torch.nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.time_proj = torch.nn.Linear(1, self.size**2)

    def unet_down_block(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            self.double_conv(in_channels, out_channels),
        )

    def double_conv(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, x, t):
        t_embed = self.time_proj(t).reshape(-1, 1, self.size, self.size)
        x = torch.cat((x, t_embed), dim=1)
        skips = []
        for i in range(self.depth):
            x = self.down[i](x)
            skips.append(x)
            x = self.maxpool(x)

        x = self.bottle_neck(x)

        for i in range(self.depth):
            x = self.up[i](x)
            x = torch.cat((x, skips.pop()), dim=1)
            x = self.up_conv[i](x)

        x = self.out(x)
        return x


if __name__ == "__main__":
    unet = unet(depth=3, in_channels=3, out_channels=3, size=32)
    input = torch.randn(16, 3, 32, 32)
    time = torch.randn(16, 1)
    output = unet(input, time)
    print(output.shape)
