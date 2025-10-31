import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Double convolution block consisting of two Conv2d-BatchNorm-ReLU sequences.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downsampling block: MaxPool2d followed by DoubleConv.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up_fU(nn.Module):
    """
    Upsampling block followed by DoubleConv. Supports bilinear interpolation or ConvTranspose2d.
    Explicitly handles channels from deep path and skip connections.
    """
    def __init__(self, in_ch, skip_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Final 1x1 convolution for output segmentation map.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Standard U-Net architecture for semantic segmentation.
    This implementation uses a 2D encoder-decoder structure with skip connections.
    Suitable for static image segmentation tasks.
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up_fU(1024, 512, 512, bilinear)
        self.up2 = Up_fU(512, 256, 256, bilinear)
        self.up3 = Up_fU(256, 128, 128, bilinear)
        self.up4 = Up_fU(128, 64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        d = self.up4(d, x1)
        return self.outc(d)

if __name__ == "__main__":
    # Test configuration
    BATCH_SIZE = 2
    CHANNELS = 20
    PATCH_SIZE = 128
    N_CLASSES_BINARY = 1
    N_CLASSES_MULTICLASS = 15
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running UNet test on {DEVICE}")

    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, PATCH_SIZE, PATCH_SIZE).to(DEVICE)

    print("\n--- Testing UNet ---")
    try:
        model_bin = UNet(n_channels=CHANNELS, n_classes=N_CLASSES_BINARY).to(DEVICE)
        output_bin = model_bin(dummy_input)
        print(f"[Binary] Input: {dummy_input.shape} -> Output: {output_bin.shape}")

        model_mc = UNet(n_channels=CHANNELS, n_classes=N_CLASSES_MULTICLASS).to(DEVICE)
        output_mc = model_mc(dummy_input)
        print(f"[Multiclass] Input: {dummy_input.shape} -> Output: {output_mc.shape}")
    except Exception as e:
        print(f"[ERROR] in UNet: {e}")