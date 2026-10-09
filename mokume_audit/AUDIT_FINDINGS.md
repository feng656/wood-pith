# Mokume U-Net Weight Audit — Findings

## Step 1: Weight Loading ✅
- 176/176 keys matched with strict loading
- 34,530,881 parameters
- Architecture: 4-layer encoder-decoder U-Net, RGB(3ch)→1ch output
- Trained for 43,980 batches

## Step 2: Architecture Ambiguity ⚠️
Multiple DoubleConv variants pass strict load (same weight shapes/keys):

| Variant | Forward | Result |
|---------|---------|--------|
| A | Conv→BN→ReLU→Conv→BN→ReLU | Narrow outputs, logits [0, 0.5] |
| B | Conv→BN→ReLU→Conv→BN (no final ReLU) | Wide outputs, logits [-5.5, 1.6] |
| C | Conv→BN→Conv→BN→ReLU | Exploding outputs (>750k) |

**Variant B is most plausible** based on output range.

## Step 3: Output Analysis ❌
- Direction correct: ring pixels have lower logits than background ✓
- Discrimination: essentially random (best F1 = 0.072 for B02, 0.012 for B01)
- Model responds to noise better than real images
- No meaningful spatial correlation with ring annotations

## Decision
**Cannot reproduce ring detection results from these weights.** The model either:
1. Was trained for a different task (not ring segmentation on these RGB images)
2. Requires preprocessing or postprocessing not discoverable from weights alone
3. Has a subtly different architecture that changes forward pass behavior

**Recommendation**: Train a new ring-only baseline from scratch.
