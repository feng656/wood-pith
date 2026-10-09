"""Step 1: Reconstruct Mokume U-Net and verify strict weight loading."""
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Conv3x3 + BN + ReLU + Conv3x3 + BN (no ReLU after second BN)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = torch.relu(x)
        return x


class UpConv(nn.Module):
    """Upsampling block: BN + bilinear upsample + Conv3x3 + BN + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.bn1(x)
        x = torch.relu(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
                                       align_corners=False)
        x = self.conv(x)
        x = self.bn2(x)
        x = torch.relu(x)
        return x


class MokumeUNet(nn.Module):
    """Reconstructed U-Net matching the 176-key checkpoint.

    Encoder: TCB1→TCB2→TCB3→TCB4→TCB5 (4× down via stride-2 first conv)
    Decoder: UC1+TCB6→UC2+TCB7→UC3+TCB8→UC4+TCB9
    Skip: TCB4→TCB6, TCB3→TCB7, TCB2→TCB8, TCB1→TCB9
    Output: conv1 (1×1) → 1 channel
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.TCB1 = DoubleConv(3, 64)
        self.TCB2 = DoubleConv(64, 128)
        self.TCB3 = DoubleConv(128, 256)
        self.TCB4 = DoubleConv(256, 512)
        self.TCB5 = DoubleConv(512, 1024)
        # Decoder upsampling
        self.UC1 = UpConv(1024, 512)
        self.UC2 = UpConv(512, 256)
        self.UC3 = UpConv(256, 128)
        self.UC4 = UpConv(128, 64)
        # Decoder double convs
        self.TCB6 = DoubleConv(1024, 512)
        self.TCB7 = DoubleConv(512, 256)
        self.TCB8 = DoubleConv(256, 128)
        self.TCB9 = DoubleConv(128, 64)
        # Output
        self.conv1 = nn.Conv2d(64, 1, 1, bias=True)
        # Pooling for encoder downsampling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder
        e1 = self.TCB1(x)          # 64ch, 1×
        e2 = self.TCB2(self.pool(e1))  # 128ch, 1/2×
        e3 = self.TCB3(self.pool(e2))  # 256ch, 1/4×
        e4 = self.TCB4(self.pool(e3))  # 512ch, 1/8×
        e5 = self.TCB5(self.pool(e4))  # 1024ch, 1/16×

        # Decoder
        d1 = self.UC1(e5)               # 512ch, 1/8×
        d1 = torch.cat([d1, e4], dim=1)  # 1024ch
        d1 = self.TCB6(d1)              # 512ch

        d2 = self.UC2(d1)               # 256ch, 1/4×
        d2 = torch.cat([d2, e3], dim=1)  # 512ch
        d2 = self.TCB7(d2)              # 256ch

        d3 = self.UC3(d2)               # 128ch, 1/2×
        d3 = torch.cat([d3, e2], dim=1)  # 256ch
        d3 = self.TCB8(d3)              # 128ch

        d4 = self.UC4(d3)               # 64ch, 1×
        d4 = torch.cat([d4, e1], dim=1)  # 128ch
        d4 = self.TCB9(d4)              # 64ch

        out = self.conv1(d4)            # 1ch
        return out


if __name__ == "__main__":
    print("Building model...")
    model = MokumeUNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nLoading checkpoint...")
    ckpt_path = "E:/dataset-clean/Mokume/unet_trained_model.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Strict load
    model.load_state_dict(state_dict, strict=True)
    print("✅ Strict load SUCCESS — all 176 keys matched!")

    # Verify key counts
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = model_keys - ckpt_keys
    extra = ckpt_keys - model_keys
    print(f"  Model keys: {len(model_keys)}")
    print(f"  Ckpt  keys: {len(ckpt_keys)}")
    print(f"  Missing: {len(missing)}, Extra: {len(extra)}")
    if missing:
        print(f"  MISSING: {list(missing)[:5]}")
    if extra:
        print(f"  EXTRA: {list(extra)[:5]}")

    print("\nModel architecture:")
    print(model)
