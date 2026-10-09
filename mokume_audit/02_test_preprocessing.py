"""Step 2: Test Mokume model with candidate preprocessing schemes.

Tests schemes A (RGB/255), B (RGB+ImageNet norm), C (BGR/255)
at input sizes 320, 512, 640. Saves outputs for visual inspection.
"""
import os, json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from mokume_audit_01_model import MokumeUNet  # from step 1

# ── Config ────────────────────────────────────────────────────
CKPT_PATH = "E:/dataset-clean/Mokume/unet_trained_model.pt"
DATA_DIR = Path("E:/dataset-clean/Mokume/MokumeDataset/MokumeDataset")
OUT_DIR = Path("D:/教务处实习/wood_preproject/树髓定位/mokume_audit/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Test on a few blocks' A-faces.
TEST_SAMPLES = ["B01/A_col.png", "B02/A_col.png", "B04/A_col.png"]

SIZES = [320, 512, 640]

SCHEMES = {
    "A_RGB_div255": {
        "color": "RGB",
        "norm": lambda t: t / 255.0,
    },
    "B_RGB_ImageNet": {
        "color": "RGB",
        "norm": lambda t: (t / 255.0 - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1))
                        / torch.tensor([0.229, 0.224, 0.225]).view(3,1,1),
    },
    "C_BGR_div255": {
        "color": "BGR",
        "norm": lambda t: t.flip(0) / 255.0,
    },
}


def load_image(path, scheme, target_size):
    """Load image, apply preprocessing, return tensor [1,3,H,W] and original size."""
    img = Image.open(path)
    orig_w, orig_h = img.size

    # Color conversion
    if scheme["color"] == "RGB":
        img = img.convert("RGB")
    elif scheme["color"] == "BGR":
        img = img.convert("RGB")
        # Will flip channels after to_tensor

    # Resize preserving aspect ratio, pad to target_size
    img_np = np.array(img, dtype=np.float32)
    if scheme["color"] == "BGR":
        img_np = img_np[..., ::-1]  # RGB -> BGR

    # Scale longest side to target_size, pad shorter side
    h, w = img_np.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Round to multiple of 16 for 4× downsampling compatibility
    new_h = ((new_h + 15) // 16) * 16
    new_w = ((new_w + 15) // 16) * 16

    # Resize
    img_resized = np.array(Image.fromarray(img_np.astype(np.uint8)).resize(
        (new_w, new_h), Image.BILINEAR
    ), dtype=np.float32)

    # Convert to tensor [3,H,W]
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1)

    # Apply normalization
    tensor = scheme["norm"](tensor)

    return tensor.unsqueeze(0), (new_h, new_w), (orig_h, orig_w), scale


@torch.no_grad()
def run_scheme(model, sample_path, scheme_name, scheme, sizes):
    """Run one preprocessing scheme across sizes."""
    results = {}
    for size in sizes:
        tensor, (h, w), (orig_h, orig_w), scale = load_image(
            sample_path, scheme, size
        )
        output = model(tensor)  # [1, 1, H, W]
        logits = output.squeeze().cpu()  # [H, W]
        probs = torch.sigmoid(logits)

        tag = f"{Path(sample_path).stem}_{scheme_name}_s{size}"
        results[size] = {
            "logits_min": float(logits.min()),
            "logits_max": float(logits.max()),
            "logits_mean": float(logits.mean()),
            "probs_min": float(probs.min()),
            "probs_max": float(probs.max()),
            "probs_mean": float(probs.mean()),
            "pct_above_03": float((probs > 0.3).float().mean()),
            "pct_above_05": float((probs > 0.5).float().mean()),
            "pct_above_07": float((probs > 0.7).float().mean()),
            "h": h, "w": w,
        }

        # Save heatmap overlay
        prob_np = (probs.numpy() * 255).astype(np.uint8)
        Image.fromarray(prob_np).save(
            OUT_DIR / f"{tag}_heatmap.png"
        )

        # Save binary at thresholds
        for th in [0.3, 0.5, 0.7]:
            binary = ((probs > th).float().numpy() * 255).astype(np.uint8)
            Image.fromarray(binary).save(
                OUT_DIR / f"{tag}_th{int(th*100):02d}.png"
            )

        # Save sigmoid distribution histogram data
        np.savez(
            OUT_DIR / f"{tag}_dist.npz",
            logits=logits.numpy().flatten(),
            probs=probs.numpy().flatten(),
        )

    return results


def main():
    print("Loading model...")
    model = MokumeUNet()
    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("Model loaded.\n")

    all_results = {}
    for sample in TEST_SAMPLES:
        sample_path = DATA_DIR / sample
        if not sample_path.exists():
            print(f"⚠ Skipping missing: {sample_path}")
            continue
        print(f"{'='*60}")
        print(f"Sample: {sample}  Size: {Image.open(sample_path).size}")
        sample_results = {}
        for scheme_name, scheme in SCHEMES.items():
            print(f"  {scheme_name}...")
            sample_results[scheme_name] = run_scheme(
                model, sample_path, scheme_name, scheme, SIZES
            )
            for size, r in sample_results[scheme_name].items():
                print(f"    s={size}: logits=[{r['logits_min']:.3f}, {r['logits_max']:.3f}] "
                      f"mean={r['logits_mean']:.3f}  "
                      f"probs>0.3={r['pct_above_03']:.3f}  "
                      f"probs>0.5={r['pct_above_05']:.3f}  "
                      f"probs>0.7={r['pct_above_07']:.3f}")
        all_results[sample] = sample_results

    # Save summary
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✅ Done. Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
