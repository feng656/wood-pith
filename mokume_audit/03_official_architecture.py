"""Step 3: Verify with the OFFICIAL architecture and preprocessing from mokumeproject-main."""
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import math
from itertools import product

# ═══════════════════════════════════════════════════════════════
# OFFICIAL Mokume U-Net architecture (from COMMON/unet.py)
# ═══════════════════════════════════════════════════════════════

class TwoConvBlock(nn.Module):
    """Conv2d → BN → LeakyReLU → Conv2d → BN → LeakyReLU"""
    def __init__(self, ch_in, ch_mid, ch_out):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_mid, kernel_size=3, padding="same")
        self.bn1   = nn.BatchNorm2d(ch_mid)
        self.rl    = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(ch_mid, ch_out, kernel_size=3, padding="same")
        self.bn2   = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x


class UpConv(nn.Module):
    """Upsample → BN → Conv2d → BN  (NO ReLU!)"""
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1  = nn.BatchNorm2d(ch_in)
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding="same")
        self.bn2  = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x


class UNet_2D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.TCB1 = TwoConvBlock(in_dim, 64, 64)
        self.TCB2 = TwoConvBlock(64, 128, 128)
        self.TCB3 = TwoConvBlock(128, 256, 256)
        self.TCB4 = TwoConvBlock(256, 512, 512)
        self.TCB5 = TwoConvBlock(512, 1024, 1024)
        self.TCB6 = TwoConvBlock(1024, 512, 512)
        self.TCB7 = TwoConvBlock(512, 256, 256)
        self.TCB8 = TwoConvBlock(256, 128, 128)
        self.TCB9 = TwoConvBlock(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.UC1 = UpConv(1024, 512)
        self.UC2 = UpConv(512, 256)
        self.UC3 = UpConv(256, 128)
        self.UC4 = UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)
        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)
        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)
        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)
        x = self.TCB5(x)

        x = self.UC1(x)
        x = self.TCB6(torch.cat([x4, x], dim=1))
        x = self.UC2(x)
        x = self.TCB7(torch.cat([x3, x], dim=1))
        x = self.UC3(x)
        x = self.TCB8(torch.cat([x2, x], dim=1))
        x = self.UC4(x)
        x = self.TCB9(torch.cat([x1, x], dim=1))
        x = self.conv1(x)
        return x


def numpy_image_to_norm_torch_data(img, PATCH_SIZE):
    """Official preprocessing: BGR/255, resize, transpose to (C,H,W)."""
    img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)
    img = np.float32(img) / 255.0
    img = img.transpose((2, 0, 1))  # (H,W,C) → (C,H,W)
    return torch.tensor(img)


def official_inference(src_img, unet, PATCH_SIZE=64):
    """Replicate the official patch-based inference with Gaussian blending."""
    H, W, _ = src_img.shape
    img = np.zeros((H, W), dtype=np.float32)
    img_count = np.zeros((H, W), dtype=np.float32)

    # Gaussian mask
    mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    cx, cy = (PATCH_SIZE - 1) / 2, (PATCH_SIZE - 1) / 2
    sigma2 = 8.0 * 8.0
    for y in range(PATCH_SIZE):
        for x in range(PATCH_SIZE):
            mask[y, x] = math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / sigma2)

    stride = PATCH_SIZE // 2  # 50% overlap

    for yi, xi in product(range(0, H, stride), range(0, W, stride)):
        y = yi if yi < H - PATCH_SIZE else H - PATCH_SIZE
        x = xi if xi < W - PATCH_SIZE else W - PATCH_SIZE
        in_patch = src_img[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :]

        in_tensor = numpy_image_to_norm_torch_data(in_patch, PATCH_SIZE)
        in_tensor = in_tensor.unsqueeze(0)  # [1, 3, 64, 64]

        with torch.no_grad():
            out_patch = unet(in_tensor)[0]  # [1, 64, 64]
        out_patch = out_patch.cpu().numpy() * 255.0  # denormalize
        out_patch = out_patch.reshape(PATCH_SIZE, PATCH_SIZE)

        img[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += mask * out_patch
        img_count[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += mask

    img = img / img_count
    img = np.clip(img, 0, 255)
    return np.uint8(img)


def main():
    CKPT_PATH = "E:/dataset-clean/Mokume/unet_trained_model.pt"
    DATA_DIR = Path("E:/dataset-clean/Mokume/MokumeDataset/MokumeDataset")
    OUT_DIR = Path("D:/教务处实习/wood_preproject/树髓定位/mokume_audit/outputs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading official model...")
    model = UNet_2D(in_dim=3, out_dim=1)
    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("✅ Official architecture — strict load SUCCESS\n")

    # Test on B01 A-face using the official pipeline
    for block in ["B01", "B02", "B04"]:
        img_path = DATA_DIR / block / "A_col.png"
        ann_path = DATA_DIR / block / "A_ann.png"

        # Load image as BGR (OpenCV default) — NO color conversion!
        src_img = cv2.imread(str(img_path))
        print(f"{block}: loaded {src_img.shape}, dtype={src_img.dtype} (BGR from cv2.imread)")

        # Resize to 256×256 (as in official 2_apply_unet.py)
        src_img = cv2.resize(src_img, (256, 256), interpolation=cv2.INTER_CUBIC)

        # Run official inference
        out_img = official_inference(src_img, model, PATCH_SIZE=64)

        # Load annotation for comparison
        ann = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)

        # Save outputs
        cv2.imwrite(str(OUT_DIR / f"{block}_official_arl.png"), out_img)

        # Statistics
        print(f"  Output: min={out_img.min()}, max={out_img.max()}, mean={out_img.mean():.1f}")
        print(f"  Annotation: shape={ann.shape if ann is not None else 'N/A'}, "
              f"unique={np.unique(ann).tolist() if ann is not None else 'N/A'}")

        # Compare with annotation: binarize output at threshold 200
        # (official code uses THRESHOLD=200 in get_image_open_contours)
        binary_out = (out_img > 200).astype(np.uint8) * 255
        cv2.imwrite(str(OUT_DIR / f"{block}_official_binary_t200.png"), binary_out)

        # For annotation: ring pixels = values != 255
        if ann is not None:
            ring_gt = (ann != 255).astype(np.uint8)
            ring_pred = (out_img > 200).astype(np.uint8)

            # Compute Dice/F1
            intersection = (ring_pred & ring_gt).sum()
            dice = 2 * intersection / (ring_pred.sum() + ring_gt.sum() + 1e-8)
            print(f"  Dice (th=200): {dice:.4f}")

    print(f"\n✅ Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
