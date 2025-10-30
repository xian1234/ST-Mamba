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
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling then double conv block.
    This version is robust and backward-compatible for both bilinear and ConvTranspose modes.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: feature map from the previous (deeper) layer
        # x2: skip connection feature map from the encoder
        x1 = self.up(x1)
        
        # Pad x1 to the size of x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

# --- Model 1: BaselineUNet (Unchanged, for reference) ---

class Up_fU(nn.Module):
    """
    Upsampling then double conv block. This version is robust and backward-compatible,
    explicitly taking channels from the deep path and the skip path.
    """
    def __init__(self, in_ch, skip_ch, out_ch, bilinear=True):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        if bilinear:
            # Override with bilinear upsampling if chosen
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
        # The convolution now correctly takes the sum of channels from the upsampled path
        # and the skip connection path.
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x1, x2):
        # x1: feature map from the previous (deeper) layer to be upsampled
        # x2: skip connection feature map from the encoder
        x1 = self.up(x1)
        
        # Pad x1 to the size of x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class BaselineUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, **kwargs):
        super(BaselineUNet, self).__init__()
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024) # Bottleneck has 1024 channels

        # Decoder using the new robust Up module
        # Up(in_ch_from_deep, skip_ch_from_encoder, out_ch_final)
        self.up1 = Up_fU(1024, 512, 512, bilinear)
        self.up2 = Up_fU(512, 256, 256, bilinear)
        self.up3 = Up_fU(256, 128, 128, bilinear)
        self.up4 = Up_fU(128, 64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)      # 64 channels
        x2 = self.down1(x1)   # 128 channels
        x3 = self.down2(x2)   # 256 channels
        x4 = self.down3(x3)   # 512 channels
        x5 = self.down4(x4)   # 1024 channels (bottleneck)

        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        d = self.up4(d, x1)
        return self.outc(d)


class Conv2plus1D(nn.Module):
    """Factorized (2+1)D convolution block."""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        super().__init__()
        # Spatial convolution (across H, W)
        self.spatial_conv = nn.Conv3d(in_channels, out_channels, (1, kernel_size[1], kernel_size[2]),
                                      padding=(0, padding[1], padding[2]), bias=False)
        # Temporal convolution (across T)
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            Conv2plus1D(in_channels, out_channels),
            Conv2plus1D(out_channels, out_channels)
        )
    def forward(self, x):
        return self.conv_block(x)

class Up3D(nn.Module):
    """Upsampling then double (2+1)D conv block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv2plus1D(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is from the decoder path, x2 is the skip connection from the encoder
        x1 = self.up(x1)
        
        # Handle potential padding differences between encoder and decoder paths
        diffT = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffT // 2, diffT - diffT // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# --- Fully Refactored UNet3D Model ---
class UNet3D(nn.Module):
    def __init__(self, n_channels=4, n_classes=1, **kwargs):
        super(UNet3D, self).__init__()

        # Encoder
        self.inc = DoubleConv2plus1D(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool3d((1, 2, 2)), DoubleConv2plus1D(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool3d((1, 2, 2)), DoubleConv2plus1D(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv2plus1D(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool3d(2), DoubleConv2plus1D(256, 512))

        # Decoder
        self.up1 = Up3D(512, 256)
        self.up2 = Up3D(256, 128)
        self.up3 = Up3D(128, 64)
        self.up4 = Up3D(64, 32)
        
        #  FIX: Final processing layers defined correctly in __init__
        # This layer collapses the temporal dimension to size 1
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        # This final convolution produces the output map
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Input shape: (B, C, T, H, W)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        d = self.up4(d, x1) # Shape: (B, 32, T, H, W)
        
        # FIX: Use the pooling layer to collapse the time dimension
        d_pooled = self.pool(d) # Shape: (B, 32, 1, H, W)
        
        # Squeeze the time dimension to make it a 2D feature map
        d_squeezed = d_pooled.squeeze(2) # Shape: (B, 32, H, W)
        
        # Final 2D convolution to get the output classes
        logits = self.outc(d_squeezed) # Shape: (B, n_classes, H, W)
        
        return logits

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=padding, bias=True)
        self.conv_can = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=True)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv_gates(combined)
        reset_gate, update_gate = torch.sigmoid(gates).chunk(2, dim=1)
        combined_can = torch.cat([x, reset_gate * h_prev], dim=1)
        candidate = torch.tanh(self.conv_can(combined_can))
        h_next = (1 - update_gate) * h_prev + update_gate * candidate
        return h_next


class SpatioTemporalUNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=1, **kwargs):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)


        self.gru1 = ConvGRUCell(input_dim=64, hidden_dim=64, kernel_size=3)
        self.gru2 = ConvGRUCell(input_dim=128, hidden_dim=128, kernel_size=3)
        self.gru3 = ConvGRUCell(input_dim=256, hidden_dim=256, kernel_size=3)
        self.gru4 = ConvGRUCell(input_dim=512, hidden_dim=512, kernel_size=3)

 
        self.up1 = Up(512 + 256, 256)
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Input shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape


        h1 = torch.zeros(B, 64, H, W, device=x.device)
        h2 = torch.zeros(B, 128, H // 2, W // 2, device=x.device)
        h3 = torch.zeros(B, 256, H // 4, W // 4, device=x.device)
        h4 = torch.zeros(B, 512, H // 8, W // 8, device=x.device)


        for t in range(T):
            xt = x[:, t, :, :, :]
            
            # Encoder path
            s1 = self.inc(xt)
            s2 = self.down1(s1)
            s3 = self.down2(s2)
            s4 = self.down3(s3)

            # Update hidden states at each level
            h1 = self.gru1(s1, h1)
            h2 = self.gru2(s2, h2)
            h3 = self.gru3(s3, h3)
            h4 = self.gru4(s4, h4)


        d = self.up1(h4, h3)
        d = self.up2(d, h2)
        d = self.up3(d, h1)
        
        return self.outc(d)



class TemporalFusionModule(nn.Module):
    def __init__(self, in_features=512, temporal_module='gru', feature_dim=512):
        super().__init__()
        self.temporal_module_type = temporal_module
        if temporal_module == 'gru':
            self.temporal_model = nn.GRU(in_features, feature_dim, num_layers=2, batch_first=True, bidirectional=True)
            self.out_features = feature_dim * 2
        elif temporal_module == 'mamba':
            if not mamba_available: raise ImportError("Mamba is not installed or available.")
            self.temporal_model = Mamba(d_model=in_features, d_state=16, d_conv=4)
            self.out_features = in_features
        else:
            raise ValueError(f"Unknown temporal module: {temporal_module}")

    def forward(self, x_seq):
        # x_seq shape: (B, T, C)
        if self.temporal_module_type == 'gru':
            fused_seq, _ = self.temporal_model(x_seq)

            return fused_seq[:, -1, :]
        else: # mamba
            fused_seq = self.temporal_model(x_seq)
            return fused_seq[:, -1, :]



class SpatioTemporalFusion(nn.Module):
    def __init__(self, n_channels=4, n_classes=1, temporal_module='mamba', patch_size=128, **kwargs):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.temporal_fusion = TemporalFusionModule(in_features=512, temporal_module=temporal_module)


        bottleneck_dim = patch_size // 8 

        self.decoder_projection = nn.Linear(self.temporal_fusion.out_features, 512 * bottleneck_dim * bottleneck_dim)
        
        self.up1 = Up(512 + 256, 256) # 
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Input shape: (B, T, C, H, W)
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

        bottleneck_sequence = torch.stack(bottleneck_vectors, dim=1) # (B, T, 512)
        v_fused = self.temporal_fusion(bottleneck_sequence) # (B, out_features)
        

        d_start = self.decoder_projection(v_fused)
        d = d_start.view(B, 512, H // 8, W // 8)
        

        d = self.up1(d, skips_last_t['s3'])
        d = self.up2(d, skips_last_t['s2'])
        d = self.up3(d, skips_last_t['s1'])
        
        return self.outc(d)



if __name__ == "__main__":
    """
    Instantiates and runs a forward pass for all models in both
    binary and multiclass configurations to check for errors.
    """
    # --- Test Configuration ---
    BATCH_SIZE = 2
    TIME_STEPS = 5
    CHANNELS = 20
    PATCH_SIZE = 128
    N_CLASSES_BINARY = 1
    N_CLASSES_MULTICLASS = 15 # An arbitrary number > 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running tests on device: {DEVICE.upper()}")

    # --- Dummy Data ---
    dummy_st_input = torch.randn(BATCH_SIZE, TIME_STEPS, CHANNELS, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
    dummy_2d_input = dummy_st_input[:, 0, :, :, :].to(DEVICE) # First timestamp for 2D model
    
    # --- Model Test Function ---
    def run_test(model_class, model_name, input_tensor, **kwargs):
        print(f"\n--- Testing Model: {model_name} ---")
        try:
            # Binary Test
            model_bin = model_class(n_channels=CHANNELS, n_classes=N_CLASSES_BINARY, **kwargs).to(DEVICE)
            output_bin = model_bin(input_tensor)
            print(f"[Binary]   Input: {input_tensor.shape} -> Output: {output_bin.shape}")

            # Multiclass Test
            model_mc = model_class(n_channels=CHANNELS, n_classes=N_CLASSES_MULTICLASS, **kwargs).to(DEVICE)
            output_mc = model_mc(input_tensor)
            print(f"[Multiclass] Input: {input_tensor.shape} -> Output: {output_mc.shape}")
        except Exception as e:
            print(f"[ERROR] in {model_name}: {e}")

    # --- Run Tests ---
    run_test(BaselineUNet, "BaselineUNet (2D)", dummy_2d_input)
    
    # UNet3D requires channel-first input: (B, C, T, H, W)
    run_test(UNet3D, "UNet3D ((2+1)D Convs)", dummy_st_input.permute(0, 2, 1, 3, 4))
    
    run_test(SpatioTemporalUNet, "SpatioTemporalUNet (ConvGRU)", dummy_st_input)
    
    # SpatioTemporalFusion has a 'temporal_module' argument
    run_test(SpatioTemporalFusion, "SpatioTemporalFusion (GRU)", dummy_st_input, 
             temporal_module='gru', patch_size=PATCH_SIZE)
             
    if mamba_available:
        run_test(SpatioTemporalFusion, "SpatioTemporalFusion (Mamba)", dummy_st_input, 
                 temporal_module='mamba', patch_size=PATCH_SIZE)
    else:
        print("\n--- Skipping Model: SpatioTemporalFusion (Mamba) ---")
        print(" Mamba is not installed. Skipping test.")

    print("\n All model tests completed. ")