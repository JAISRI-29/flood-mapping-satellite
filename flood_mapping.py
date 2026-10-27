"""
Flood Mapping Using Satellite Images

Flood-water segmentation from Sentinel-1 SAR imagery
using the Sen1Floods11 dataset.
if __name__ == "__main__":
    main()
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
