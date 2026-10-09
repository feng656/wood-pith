"""
训练 CNN 模型从 patch 图片直接回归髓心坐标
==========================================
输入: patch 图片 (224x224 RGB)
输出: (cx, cy) 髓心在 patch 内的像素坐标

模型: 轻量自定义 CNN (~1.5M params, CPU 友好)
训练: MSE loss, Adam, batch_size=32, ~50 epochs
"""
import json, cv2, numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==================== 配置 ====================
PATCH_ROOT = Path(r"D:\教务处实习\木材数据1\patch_dataset\裁剪矩形")
OUT_DIR = Path(r"D:\教务处实习\木材数据1\patch_dataset\调试验证定位")
MODEL_DIR = OUT_DIR / "cnn_model"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-3
IMG_SIZE = 224

print(f"设备: {DEVICE}")

# ==================== 加载数据 ====================
ALL_JSONS = sorted([j for j in PATCH_ROOT.glob("*/*.json") if not j.stem.endswith("_新髓心")])
print(f"加载 JSON 标注...")

samples = []
for jf in ALL_JSONS:
    with open(jf, 'r', encoding='utf-8') as f:
        d = json.load(f)
    img_path = jf.parent / f"{d['patch_name']}.jpg"
    if img_path.exists():
        samples.append({
            'img_path': str(img_path),
            'cx': d['cx_local'],
            'cy': d['cy_local'],
            'size': d['size'],
            'pith_in_patch': d['pith_in_patch'],
            'patch_name': d['patch_name'],
        })

print(f"有效样本: {len(samples)}")

# ==================== 数据划分 ====================
# 按样本(sample)分组划分, 避免同一样本的不同patch出现在训练/测试中
# 简单起见先用随机划分
train_data, test_data = train_test_split(samples, test_size=0.15, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.18, random_state=42)
print(f"训练: {len(train_data)}  验证: {len(val_data)}  测试: {len(test_data)}")

# ==================== Dataset ====================
class PithDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        img = cv2.imread(d['img_path'])
        if img is None:
            return self.__getitem__((idx + 1) % len(self.data))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # 归一化图像
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.transpose(2, 0, 1)  # CHW

        # 目标: 髓心坐标归一化到 [-5, 5]
        # cx_local 可以是负值(髓心在块外), 除以 size/2 归一化
        half = d['size'] / 2
        tx = d['cx'] / (half + 1e-8)
        ty = d['cy'] / (half + 1e-8)

        # Clamp
        tx = np.clip(tx, -8, 8)
        ty = np.clip(ty, -8, 8)

        return torch.FloatTensor(img), torch.FloatTensor([tx, ty])

# ==================== 模型 ====================
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
                nn.BatchNorm2d(out_ch),
            )

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
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResidualBlock(32, 64), ResidualBlock(64, 64))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256))
        self.layer4 = nn.Sequential(ResidualBlock(256, 512, stride=2), ResidualBlock(512, 512))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ==================== 训练 ====================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, n = 0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        preds = model(imgs)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, n = 0, 0
    errors = []
    for imgs, targets in loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        preds = model(imgs)
        loss = criterion(preds, targets)
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        # 反归一化误差 (px)
        half = IMG_SIZE / 2  # 使用IMG_SIZE作为近似, 实际推理时用真实size
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            err = np.sqrt((p[0]-t[0])**2 + (p[1]-t[1])**2) * half
            errors.append(err)
    return total_loss / n, np.median(errors)

print(f"\n{'='*60}")
print(f"开始训练 PithCNN...")
print(f"参数量: {sum(p.numel() for p in PithCNN().parameters())/1e6:.1f}M")
print(f"Epochs: {EPOCHS}  Batch: {BATCH_SIZE}  LR: {LR}")

train_loader = DataLoader(PithDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(PithDataset(val_data), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(PithDataset(test_data), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = PithCNN().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()

best_val_err = float('inf')
history = []

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_err = eval_epoch(model, val_loader, criterion)
    scheduler.step()

    history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_err_px': val_err})

    if val_err < best_val_err:
        best_val_err = val_err
        torch.save(model.state_dict(), str(MODEL_DIR / 'best_model.pt'))

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Err: {val_err:.0f}px")

# ==================== 测试 ====================
print(f"\n{'='*60}")
print(f"测试集评估...")
model.load_state_dict(torch.load(str(MODEL_DIR / 'best_model.pt'), weights_only=True))
test_loss, test_err = eval_epoch(model, test_loader, criterion)

# 更详细的测试评估
@torch.no_grad()
def detailed_test(model, loader, data_list):
    model.eval()
    results = []
    for idx, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        preds = model(imgs).cpu().numpy()

        batch_data = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
        for p, t, d in zip(preds, targets.numpy(), batch_data):
            half = d['size'] / 2
            cx_pred = p[0] * half
            cy_pred = p[1] * half
            cx_gt = d['cx']
            cy_gt = d['cy']
            err = np.sqrt((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2)
            results.append({
                'patch_name': d['patch_name'],
                'gt_cx': cx_gt, 'gt_cy': cy_gt,
                'pred_cx': cx_pred, 'pred_cy': cy_pred,
                'error': err,
                'size': d['size'],
            })

    return pd.DataFrame(results)

test_df = detailed_test(model, test_loader, test_data)
test_df.to_csv(MODEL_DIR / 'test_predictions.csv', index=False)

h4_df = pd.read_csv(OUT_DIR / "final_all_methods.csv")
h4_test = h4_df[h4_df['patch_name'].isin(test_df['patch_name'])]

print(f"\n测试集结果 ({len(test_df)} 块):")
print(f"  CNN 中位数误差: {test_df['error'].median():.0f} px")
print(f"  CNN 均值误差:   {test_df['error'].mean():.0f} px")
print(f"  H4  中位数误差: {h4_test['H4_err'].median():.0f} px")
print(f"  A   中位数误差: {h4_test['A_err'].median():.0f} px")
print(f"  B   中位数误差: {h4_test['B_err'].median():.0f} px")

# ==================== 保存 ====================
history_df = pd.DataFrame(history)
history_df.to_csv(MODEL_DIR / 'training_history.csv', index=False)

# 保存模型信息
model_info = {
    'architecture': 'PithCNN',
    'params': sum(p.numel() for p in model.parameters()),
    'input_size': IMG_SIZE,
    'output': '(cx_normalized, cy_normalized)',
    'normalization': 'cx / (patch_size/2), clipped to [-8, 8]',
    'train_samples': len(train_data),
    'val_samples': len(val_data),
    'test_samples': len(test_data),
    'best_val_err_px': best_val_err,
    'test_err_median': float(test_df['error'].median()),
    'test_err_mean': float(test_df['error'].mean()),
}

with open(MODEL_DIR / 'model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, indent=2, ensure_ascii=False)

print(f"\n模型已保存到: {MODEL_DIR}")
print(f"  best_model.pt  test_predictions.csv  training_history.csv  model_info.json")

# ==================== 对比总结 ====================
print(f"\n{'='*60}")
print(f"方法与模型对比 (测试集, n={len(test_df)})")
print(f"{'方法':<25} {'中位数(px)':<12} {'均值(px)':<12}")
print('-'*50)
for name, vals in [
    ('A_独立圆拟合', h4_test['A_err']),
    ('B_线性同心回归', h4_test['B_err']),
    ('H4_几何混合', h4_test['H4_err']),
    ('CNN_学习模型', test_df['error']),
]:
    print(f"{name:<25} {np.median(vals):<12.0f} {np.mean(vals):<12.0f}")

print(f"\n完成!")
