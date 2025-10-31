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

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for capturing spatio-temporal dependencies.
    Implements standard LSTM gates with convolutional operations.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=True)

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        input_gate, forget_gate, output_gate, cell_gate = gates.chunk(4, dim=1)
        i = torch.sigmoid(input_gate)
        f = torch.sigmoid(forget_gate)
        o = torch.sigmoid(output_gate)
        g = torch.tanh(cell_gate)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)

class ConvLSTMUNet(nn.Module):
    """
    Spatio-temporal U-Net with Convolutional LSTM cells integrated at each encoder level.
    Similar to ConvGRUUNet but uses LSTM for temporal modeling. (ViT integration not included as per source code;
    this serves as a baseline ConvLSTM variant for comparison in heritage monitoring tasks.)
    Processes sequences (B, T, C, H, W) by applying LSTMs over time at feature levels.
    """
    def __init__(self, n_channels=4, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.lstm1 = ConvLSTMCell(64, 64, 3)
        self.lstm2 = ConvLSTMCell(128, 128, 3)
        self.lstm3 = ConvLSTMCell(256, 256, 3)
        self.lstm4 = ConvLSTMCell(512, 512, 3)
        self.up1 = Up(512 + 256, 256)
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Expected input: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        state1 = (torch.zeros(B, 64, H, W, device=x.device), torch.zeros(B, 64, H, W, device=x.device))
        state2 = (torch.zeros(B, 128, H // 2, W // 2, device=x.device), torch.zeros(B, 128, H // 2, W // 2, device=x.device))
        state3 = (torch.zeros(B, 256, H // 4, W // 4, device=x.device), torch.zeros(B, 256, H // 4, W // 4, device=x.device))
        state4 = (torch.zeros(B, 512, H // 8, W // 8, device=x.device), torch.zeros(B, 512, H // 8, W // 8, device=x.device))
        for t in range(T):
            xt = x[:, t, :, :, :]
            s1 = self.inc(xt)
            s2 = self.down1(s1)
            s3 = self.down2(s2)
            s4 = self.down3(s3)
            _, state1 = self.lstm1(s1, state1)
            _, state2 = self.lstm2(s2, state2)
            _, state3 = self.lstm3(s3, state3)
            _, state4 = self.lstm4(s4, state4)
        h1, _ = state1
        h2, _ = state2
        h3, _ = state3
        h4, _ = state4
        d = self.up1(h4, h3)
        d = self.up2(d, h2)
        d = self.up3(d, h1)
        return self.outc(d)

if __name__ == "__main__":
    # Test configuration
    BATCH_SIZE = 2
    TIME_STEPS = 5
    CHANNELS = 20
    PATCH_SIZE = 128
    N_CLASSES_BINARY = 1
    N_CLASSES_MULTICLASS = 15
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running ConvLSTMUNet test on {DEVICE}")

    dummy_input = torch.randn(BATCH_SIZE, TIME_STEPS, CHANNELS, PATCH_SIZE, PATCH_SIZE).to(DEVICE)

    print("\n--- Testing ConvLSTMUNet ---")
    try:
        model_bin = ConvLSTMUNet(n_channels=CHANNELS, n_classes=N_CLASSES_BINARY).to(DEVICE)
        output_bin = model_bin(dummy_input)
        print(f"[Binary] Input: {dummy_input.shape} -> Output: {output_bin.shape}")

        model_mc = ConvLSTMUNet(n_channels=CHANNELS, n_classes=N_CLASSES_MULTICLASS).to(DEVICE)
        output_mc = model_mc(dummy_input)
        print(f"[Multiclass] Input: {dummy_input.shape} -> Output: {output_mc.shape}")
    except Exception as e:
        print(f"[ERROR] in ConvLSTMUNet: {e}")