"""加载训练好的CNN模型并做全量评测"""
import json, cv2, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time

# ==================== 路径 ====================
PATCH_ROOT = Path(r"D:\教务处实习\木材数据1\patch_dataset\裁剪矩形")
MODEL_DIR = Path(r"D:\教务处实习\木材数据1\patch_dataset\调试验证定位\cnn_model")
OUT_DIR = Path(r"D:\教务处实习\木材数据1\patch_dataset\调试验证定位")
DEVICE = torch.device('cpu')
IMG_SIZE = 224
BATCH_SIZE = 64

# ==================== 模型定义 ====================
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class PithCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = nn.Sequential(ResidualBlock(32, 64), ResidualBlock(64, 64))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256))
        self.layer4 = nn.Sequential(ResidualBlock(256, 512, stride=2), ResidualBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 2))

    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x); x = x.view(x.size(0), -1)
        return self.fc(x)

# ==================== 加载模型 ====================
model = PithCNN()
model.load_state_dict(torch.load(str(MODEL_DIR / 'best_model.pt'), weights_only=True))
model.to(DEVICE)
model.eval()
print(f"模型已加载, 参数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ==================== 加载全部数据 ====================
ALL_JSONS = sorted([j for j in PATCH_ROOT.glob("*/*.json") if not j.stem.endswith("_新髓心")])
print(f"加载 {len(ALL_JSONS)} 个样本...")

samples = []
for jf in ALL_JSONS:
    with open(jf, 'r', encoding='utf-8') as f:
        d = json.load(f)
    img_path = jf.parent / f"{d['patch_name']}.jpg"
    if img_path.exists():
        samples.append({
            'img_path': str(img_path), 'cx': d['cx_local'], 'cy': d['cy_local'],
            'size': d['size'], 'pith_in_patch': d['pith_in_patch'],
            'patch_name': d['patch_name'], 'sample': d['sample'],
            'dist_to_pith': d['dist_to_pith'], 'angle_span_deg': d['angle_span_deg'],
            'n_rings': d['n_ring_arcs'],
        })

print(f"有效: {len(samples)}")

# ==================== 全量推理 ====================
results = []
t0 = time.time()

for idx, d in enumerate(samples):
    img = cv2.imread(d['img_path'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img).cpu().numpy()[0]

    half = d['size'] / 2
    cx_pred = float(pred[0] * half)
    cy_pred = float(pred[1] * half)
    err = np.sqrt((cx_pred - d['cx'])**2 + (cy_pred - d['cy'])**2)

    results.append({
        'patch_name': d['patch_name'], 'sample': d['sample'],
        'size': d['size'], 'gt_cx': d['cx'], 'gt_cy': d['cy'],
        'cnn_cx': cx_pred, 'cnn_cy': cy_pred, 'cnn_err': err,
        'pith_in_patch': d['pith_in_patch'], 'dist_to_pith': d['dist_to_pith'],
        'angle_span_deg': d['angle_span_deg'], 'n_rings': d['n_rings'],
    })

    if (idx+1) % 1000 == 0:
        df_temp = pd.DataFrame(results)
        print(f"  进度: {idx+1}/{len(samples)}  CNN中位数={df_temp['cnn_err'].median():.0f}px")

dt = time.time() - t0
df = pd.DataFrame(results)
print(f"推理完成! 耗时 {dt:.0f}s ({len(samples)/dt:.1f} 样本/秒)")

# ==================== 对比 H4 ====================
h4_df = pd.read_csv(OUT_DIR / "final_all_methods.csv")
merged = df.merge(h4_df[['patch_name', 'A_err', 'B_err', 'H4_err']], on='patch_name', how='left')

print(f"\n{'='*80}")
print(f"全量对比 (n={len(merged)})")
print(f"{'方法':<25} {'中位数':<10} {'均值':<10} {'<200':<8} {'<500':<8} {'<1000':<8}")
print('-'*65)
for name, col in [('A_独立圆拟合', 'A_err'), ('B_线性同心回归', 'B_err'),
                   ('H4_几何混合', 'H4_err'), ('CNN_学习模型', 'cnn_err')]:
    errs = merged[col].dropna()
    print(f"{name:<25} {np.median(errs):<10.0f} {np.mean(errs):<10.0f} "
          f"{100*(errs<200).mean():<8.0f} {100*(errs<500).mean():<8.0f} {100*(errs<1000).mean():<8.0f}")

# ==================== 分组对比 ====================
print(f"\n{'='*80}")
print(f"分组中位数对比")
groups = [
    ('全部', slice(None)),
    ('pith_IN', merged['pith_in_patch']==True),
    ('pith_OUT', merged['pith_in_patch']==False),
    ('dist<500', merged['dist_to_pith']<500),
    ('dist>1500', merged['dist_to_pith']>=1500),
    ('ang<45', merged['angle_span_deg']<45),
    ('ang>90', merged['angle_span_deg']>=90),
]
print(f"{'分组':<16} {'N':<6} {'A':<10} {'B':<10} {'H4':<10} {'CNN':<10}")
print('-'*60)
for label, cond in groups:
    sub = merged[cond]
    print(f"{label:<16} {len(sub):<6} {sub['A_err'].median():<10.0f} "
          f"{sub['B_err'].median():<10.0f} {sub['H4_err'].median():<10.0f} {sub['cnn_err'].median():<10.0f}")

# ==================== 保存 ====================
df.to_csv(MODEL_DIR / 'cnn_all_predictions.csv', index=False, encoding='utf-8-sig')
merged.to_csv(MODEL_DIR / 'cnn_vs_h4_comparison.csv', index=False, encoding='utf-8-sig')

model_info = {
    'architecture': 'PithCNN', 'params': sum(p.numel() for p in model.parameters()),
    'input_size': IMG_SIZE, 'total_samples': len(samples),
    'median_error': float(df['cnn_err'].median()), 'mean_error': float(df['cnn_err'].mean()),
    'inference_speed': f'{len(samples)/dt:.1f} samples/sec',
}
with open(MODEL_DIR / 'model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, indent=2, ensure_ascii=False)

print(f"\n输出已保存: {MODEL_DIR}")
print(f"  cnn_all_predictions.csv  cnn_vs_h4_comparison.csv  model_info.json")
