"""
Flood Mapping Using Satellite Images

Flood-water segmentation from Sentinel-1 SAR imagery
using the Sen1Floods11 dataset.
"""
import argparse
import os
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader, Dataset
class Sen1Floods11:
    """Paths and loaders for the hand-labelled Sen1Floods11 subset."""

    def __init__(self, root):
        base = f"{root}/v1.1"
        self.s1_dir = f"{base}/data/flood_events/HandLabeled/S1Hand"
        self.lb_dir = f"{base}/data/flood_events/HandLabeled/LabelHand"
        self.split_dir = f"{base}/splits/flood_handlabeled"

    def split(self, name):  # 'train' | 'valid' | 'test'
        df = pd.read_csv(f"{self.split_dir}/flood_{name}_data.csv", header=None)
        return df.values.tolist()  # [[s1_file, label_file], ...]

    def load_pair(self, s1_name, lb_name):
        with rasterio.open(f"{self.s1_dir}/{s1_name}") as f:
            x = f.read().astype(np.float32)          # (2, H, W): VV, VH in dB
        with rasterio.open(f"{self.lb_dir}/{lb_name}") as f:
            y = f.read(1).astype(np.int64)           # 1 water, 0 land, -1 no data
        x = np.nan_to_num(x, nan=-50.0)
        x = (np.clip(x, -30, 5) + 30) / 35.0         # scale dB to [0, 1]
        return x, y


class FloodDataset(Dataset):
    def __init__(self, data, files, train=False, crop=256):
        self.data, self.files, self.train, self.crop = data, files, train, crop

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x, y = self.data.load_pair(*self.files[i])
        if self.train:
            h = np.random.randint(0, x.shape[1] - self.crop + 1)
            w = np.random.randint(0, x.shape[2] - self.crop + 1)
            x, y = x[:, h:h + self.crop, w:w + self.crop], y[h:h + self.crop, w:w + self.crop]
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def metrics(pred, y):
    """IoU / F1 / precision / recall over valid (label >= 0) pixels."""
    valid = y >= 0
    p, t = pred[valid].astype(bool), (y[valid] == 1)
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    eps = 1e-9
    return {"IoU": tp / (tp + fp + fn + eps),
            "F1": 2 * tp / (2 * tp + fp + fn + eps),
            "Precision": tp / (tp + fp + eps),
            "Recall": tp / (tp + fn + eps)}


def average(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def eval_baseline(data, files, mode):
    """Water is dark in SAR. mode: 'fixed' (VV < -15 dB) or 'otsu' (per-image)."""
    rows = []
    for s1, lb in files:
        x, y = data.load_pair(s1, lb)
        vv_db = x[0] * 35 - 30
        thr = -15.0 if mode == "fixed" else threshold_otsu(vv_db[y >= 0])
        rows.append(metrics(vv_db < thr, y))
    return average(rows)


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, in_ch=2, base=32, n_classes=2):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3 = block(in_ch, c[0]), block(c[0], c[1]), block(c[1], c[2])
        self.bottleneck = block(c[2], c[3])
        self.up3, self.d3 = nn.ConvTranspose2d(c[3], c[2], 2, 2), block(c[3], c[2])
        self.up2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), block(c[2], c[1])
        self.up1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), block(c[1], c[0])
        self.out = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.up3(b), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows, loss_sum = [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, ignore_index=-1).item()
        pred = out.argmax(1).cpu().numpy()
        rows += [metrics(p, t) for p, t in zip(pred, y.cpu().numpy())]
    return loss_sum / len(loader), average(rows)


def plot_samples(data, files, out):
    fig, ax = plt.subplots(3, 4, figsize=(14, 10))
    for r, i in enumerate(random.sample(range(len(files)), 3)):
        x, y = data.load_pair(*files[i])
        ax[r, 0].imshow(x[0], cmap="gray"); ax[r, 0].set_title("VV")
        ax[r, 1].imshow(x[1], cmap="gray"); ax[r, 1].set_title("VH")
        ax[r, 2].imshow(np.where(y < 0, np.nan, y), cmap="Blues"); ax[r, 2].set_title("Label (water=1)")
        ax[r, 3].hist(x[0].ravel(), bins=60); ax[r, 3].set_title("VV histogram")
        for k in range(3):
            ax[r, k].axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/01_samples.png", dpi=150); plt.close()


def plot_curves(hist, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.epoch, hist.train_loss, label="train")
    ax[0].plot(hist.epoch, hist.val_loss, label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist.epoch, hist.val_iou); ax[1].set_title("Validation IoU"); ax[1].set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(f"{out}/02_training_curves.png", dpi=150); plt.close()


def plot_predictions(data, files, model, device, out):
    model.eval()
    fig, ax = plt.subplots(4, 4, figsize=(14, 14))
    for r, i in enumerate(random.sample(range(len(files)), 4)):
        x, y = data.load_pair(*files[i])
        with torch.no_grad():
            p = model(torch.from_numpy(x[None]).to(device)).argmax(1)[0].cpu().numpy()
        err = np.zeros((*y.shape, 3))
        err[(p == 1) & (y == 1)] = [0, .6, 1]
        err[(p == 1) & (y == 0)] = [1, 0, 0]
        err[(p == 0) & (y == 1)] = [1, .8, 0]
        ax[r, 0].imshow(x[0], cmap="gray"); ax[r, 0].set_title("SAR (VV)")
        ax[r, 1].imshow(np.where(y < 0, np.nan, y), cmap="Blues"); ax[r, 1].set_title("Ground truth")
        ax[r, 2].imshow(p, cmap="Blues"); ax[r, 2].set_title("U-Net prediction")
        ax[r, 3].imshow(err); ax[r, 3].set_title("Blue=TP  Red=FP  Yellow=FN")
        for k in range(4):
            ax[r, k].axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/04_predictions.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser(description="Flood Mapping Using Satellite Images")
    ap.add_argument("--data-root", default="data/sen1floods11")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    print("Device:", device)

    data = Sen1Floods11(a.data_root)
    train_f, val_f, test_f = data.split("train"), data.split("valid"), data.split("test")
    print(f"train {len(train_f)} | val {len(val_f)} | test {len(test_f)}")

    train_dl = DataLoader(FloodDataset(data, train_f, train=True), batch_size=a.batch_size,
                          shuffle=True, num_workers=2)
    val_dl = DataLoader(FloodDataset(data, val_f), batch_size=4)
    test_dl = DataLoader(FloodDataset(data, test_f), batch_size=4)

    plot_samples(data, train_f, a.out)

    # Baselines
    results = {"Fixed VV < -15 dB": eval_baseline(data, test_f, "fixed"),
               "Otsu on VV": eval_baseline(data, test_f, "otsu")}

    # Train U-Net
    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    history, best = [], 0.0
    for ep in range(1, a.epochs + 1):
        model.train()
        tl = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, ignore_index=-1)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()
        vl, vm = evaluate(model, val_dl, device)
        history.append({"epoch": ep, "train_loss": tl / len(train_dl), "val_loss": vl, "val_iou": vm["IoU"]})
        if vm["IoU"] > best:
            best = vm["IoU"]
            torch.save(model.state_dict(), f"{a.out}/unet_best.pt")
        print(f"epoch {ep:02d}  train {tl/len(train_dl):.4f}  val {vl:.4f}  IoU {vm['IoU']:.3f}")
    hist = pd.DataFrame(history)
    hist.to_csv(f"{a.out}/history.csv", index=False)
    plot_curves(hist, a.out)

    # Test
    model.load_state_dict(torch.load(f"{a.out}/unet_best.pt", map_location=device))
    results["U-Net"] = evaluate(model, test_dl, device)[1]
    table = pd.DataFrame(results).T.round(3)
    table.to_csv(f"{a.out}/test_metrics.csv")
    print("\nTest metrics:\n", table)

    table[["IoU", "F1"]].plot.bar(figsize=(7, 4), rot=15, title="Test performance by method")
    plt.ylim(0, 1); plt.tight_layout(); plt.savefig(f"{a.out}/03_method_comparison.png", dpi=150); plt.close()
    plot_predictions(data, test_f, model, device, a.out)
    print(f"\nDone. Figures and metrics saved in {a.out}/")


if __name__ == "__main__":
    main()
    
