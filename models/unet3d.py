import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2plus1D(nn.Module):
    """
    Factorized (2+1)D convolution: spatial Conv3d followed by temporal Conv3d.
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        super().__init__()
        self.spatial_conv = nn.Conv3d(in_channels, out_channels, (1, kernel_size[1], kernel_size[2]),
                                      padding=(0, padding[1], padding[2]), bias=False)
        self.temporal_conv = nn.Conv3d(out_channels, out_channels, (kernel_size[0], 1, 1),
                                       padding=(padding[0], 0, 0), bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = self.bn(x)
        return self.relu(x)

class DoubleConv2plus1D(nn.Module):
    """
    Double (2+1)D convolution block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            Conv2plus1D(in_channels, out_channels),
            Conv2plus1D(out_channels, out_channels)
        )

    def forward(self, x):
        return self.conv_block(x)

class Up3D(nn.Module):
    """
    3D upsampling block followed by DoubleConv2plus1D.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv2plus1D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffT = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffT // 2, diffT - diffT // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    """
    3D U-Net architecture using (2+1)D convolutions for spatio-temporal segmentation.
    Processes input as (B, C, T, H, W) and outputs a 2D segmentation map by averaging over time.
    """
    def __init__(self, n_channels=4, n_classes=1):
        super(UNet3D, self).__init__()
        self.inc = DoubleConv2plus1D(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool3d((1, 2, 2)), DoubleConv2plus1D(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool3d((1, 2, 2)), DoubleConv2plus1D(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv2plus1D(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool3d(2), DoubleConv2plus1D(256, 512))
        self.up1 = Up3D(512, 256)
        self.up2 = Up3D(256, 128)
        self.up3 = Up3D(128, 64)
        self.up4 = Up3D(64, 32)
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Expected input: (B, C, T, H, W)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        d = self.up4(d, x1)
        d_pooled = self.pool(d)
        d_squeezed = d_pooled.squeeze(2)
        logits = self.outc(d_squeezed)
        return logits

if __name__ == "__main__":
    # Test configuration
    BATCH_SIZE = 2
    TIME_STEPS = 5
    CHANNELS = 20
    PATCH_SIZE = 128
    N_CLASSES_BINARY = 1
    N_CLASSES_MULTICLASS = 15
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running UNet3D test on {DEVICE}")

    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, TIME_STEPS, PATCH_SIZE, PATCH_SIZE).to(DEVICE)

    print("\n--- Testing UNet3D ---")
    try:
        model_bin = UNet3D(n_channels=CHANNELS, n_classes=N_CLASSES_BINARY).to(DEVICE)
        output_bin = model_bin(dummy_input)
        print(f"[Binary] Input: {dummy_input.shape} -> Output: {output_bin.shape}")

        model_mc = UNet3D(n_channels=CHANNELS, n_classes=N_CLASSES_MULTICLASS).to(DEVICE)
        output_mc = model_mc(dummy_input)
        print(f"[Multiclass] Input: {dummy_input.shape} -> Output: {output_mc.shape}")
    except Exception as e:
        print(f"[ERROR] in UNet3D: {e}")