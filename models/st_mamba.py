import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    mamba_available = True
except ImportError:
    mamba_available = False
    Mamba = None

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

class Up(nn.Module):
    """
    Upsampling block followed by DoubleConv. Supports bilinear interpolation or ConvTranspose2d.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

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

class TemporalFusionModule(nn.Module):
    """
    Temporal fusion module using Mamba for efficient long-sequence modeling.
    Requires mamba_ssm library; raises error if unavailable.
    """
    def __init__(self, in_features=512):
        super().__init__()
        if not mamba_available:
            raise ImportError("Mamba is not installed or available. Please install mamba_ssm.")
        self.temporal_model = Mamba(d_model=in_features, d_state=16, d_conv=4)
        self.out_features = in_features

    def forward(self, x_seq):
        fused_seq = self.temporal_model(x_seq)
        return fused_seq[:, -1, :]

class STMamba(nn.Module):
    """
    Spatio-Temporal Mamba model for multi-decadal heritage monitoring.
    Decouples static context (spatial features from last timestep) and dynamic changes (temporal fusion via Mamba).
    Input: (B, T, C, H, W); Output: (B, n_classes, H, W).
    """
    def __init__(self, n_channels=4, n_classes=1, patch_size=128):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.temporal_fusion = TemporalFusionModule(in_features=512)
        bottleneck_dim = patch_size // 8
        self.decoder_projection = nn.Linear(self.temporal_fusion.out_features, 512 * bottleneck_dim * bottleneck_dim)
        self.up1 = Up(512 + 256, 256)
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Expected input: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        bottleneck_vectors = []
        skips_last_t = {}
        for t in range(T):
            xt = x[:, t, :, :, :]
            s1 = self.inc(xt)
            s2 = self.down1(s1)
            s3 = self.down2(s2)
            s4 = self.down3(s3)
            v = self.pool(s4).view(B, -1)
            bottleneck_vectors.append(v)
            if t == T - 1:
                skips_last_t['s1'] = s1
                skips_last_t['s2'] = s2
                skips_last_t['s3'] = s3
        bottleneck_sequence = torch.stack(bottleneck_vectors, dim=1)  # (B, T, 512)
        v_fused = self.temporal_fusion(bottleneck_sequence)  # (B, out_features)
        d_start = self.decoder_projection(v_fused)
        d = d_start.view(B, 512, H // 8, W // 8)
        d = self.up1(d, skips_last_t['s3'])
        d = self.up2(d, skips_last_t['s2'])
        d = self.up3(d, skips_last_t['s1'])
        return self.outc(d)

if __name__ == "__main__":
    if not mamba_available:
        print("Mamba is not available. Skipping STMamba test.")
    else:
        # Test configuration
        BATCH_SIZE = 2
        TIME_STEPS = 5
        CHANNELS = 20
        PATCH_SIZE = 128
        N_CLASSES_BINARY = 1
        N_CLASSES_MULTICLASS = 15
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Running STMamba test on {DEVICE}")

        dummy_input = torch.randn(BATCH_SIZE, TIME_STEPS, CHANNELS, PATCH_SIZE, PATCH_SIZE).to(DEVICE)

        print("\n--- Testing STMamba ---")
        try:
            model_bin = STMamba(n_channels=CHANNELS, n_classes=N_CLASSES_BINARY, patch_size=PATCH_SIZE).to(DEVICE)
            output_bin = model_bin(dummy_input)
            print(f"[Binary] Input: {dummy_input.shape} -> Output: {output_bin.shape}")

            model_mc = STMamba(n_channels=CHANNELS, n_classes=N_CLASSES_MULTICLASS, patch_size=PATCH_SIZE).to(DEVICE)
            output_mc = model_mc(dummy_input)
            print(f"[Multiclass] Input: {dummy_input.shape} -> Output: {output_mc.shape}")
        except Exception as e:
            print(f"[ERROR] in STMamba: {e}")