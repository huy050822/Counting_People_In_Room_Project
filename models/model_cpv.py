import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from scipy.ndimage import gaussian_filter

from models.data_path import Data_Path

DEVICE = Data_Path.DEVICE

class DensityMapGenerator:
    def __init__(self, sigma=4):
        self.sigma = sigma

    def generate(self, shape, points):
        h, w = shape
        density = np.zeros((h, w), dtype=np.float32)

        if len(points) == 0:
            return density

        for x, y in points:
            x = min(w-1, max(0, int(x)))
            y = min(h-1, max(0, int(y)))
            density[y, x] = 1

        density = gaussian_filter(density, sigma=self.sigma)

        if density.sum() > 0:
            density = density / density.sum() * len(points)

        return density

class GTGenerator:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.dm_generator = DensityMapGenerator()

    def create_gt(self, img_dir, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        for img_name in tqdm(os.listdir(img_dir)):
            if not img_name.endswith(".jpg"):
                continue

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            h, w = img.shape[:2]

            points = self.df[self.df['FILENAME'] == img_name][['X','Y']].values

            density = self.dm_generator.generate((h, w), points)

            save_path = os.path.join(save_dir, img_name.replace(".jpg", ".npy"))
            np.save(save_path, density)

# ===== MODEL =====
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer2 = self._make_layer(64, 64, 2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        self.layer5 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = [ResidualBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class DensityHead(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        return F.relu(self.head(x))

class CrowdCountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNetBackbone()
        self.head = DensityHead(512)

    def forward(self, x):
        return self.head(self.backbone(x))

class CrowdDataset(Dataset):
    def __init__(self, img_dir, gt_dir):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        img = self.transform(img)

        density = np.load(
            os.path.join(self.gt_dir, img_name.replace(".jpg", ".npy"))
        )

        h, w = density.shape

        density = cv2.resize(density, (7, 7), interpolation=cv2.INTER_AREA)

        if density.sum() > 0:
            density = density * ((h * w) / (7 * 7))

        density = torch.tensor(density).float().unsqueeze(0)

        return img, density

class Metrics:
    @staticmethod
    def compute(pred, gt):
        pred, gt = np.array(pred), np.array(gt)

        mae = np.mean(np.abs(pred - gt))
        mse = np.mean((pred - gt)**2)

        ss_res = np.sum((gt - pred)**2)
        ss_tot = np.sum((gt - np.mean(gt))**2)

        r2 = 1 - ss_res / (ss_tot + 1e-8)

        return mae, mse, r2

# ===== TRAINER =====
class Trainer:
    def __init__(self, train_dir, val_dir, test_dir,
                 gt_train, gt_val, gt_test):

        self.device = DEVICE
        self.best_mae = float("inf")
        self.save_path = "trained_model.pth"

        self.train_loader = DataLoader(
            CrowdDataset(train_dir, gt_train),
            batch_size=8, shuffle=True, num_workers=2, pin_memory=True
        )

        self.val_loader = DataLoader(
            CrowdDataset(val_dir, gt_val),
            batch_size=8
        )

        self.test_loader = DataLoader(
            CrowdDataset(test_dir, gt_test),
            batch_size=8
        )

        self.model = CrowdCountingModel().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.mse = nn.MSELoss()
        self.l1  = nn.L1Loss()

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0

        for imgs, gts in self.train_loader:
            imgs, gts = imgs.to(self.device), gts.to(self.device)

            preds = self.model(imgs)

            loss = self.mse(preds, gts) + 0.1 * self.l1(preds, gts)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()

        preds_list, gts_list = [], []

        with torch.no_grad():
            for imgs, gts in loader:
                imgs, gts = imgs.to(self.device), gts.to(self.device)

                preds = self.model(imgs)

                preds_count = preds.sum(dim=[1,2,3])
                gts_count = gts.sum(dim=[1,2,3])

                preds_list.extend(preds_count.cpu().numpy())
                gts_list.extend(gts_count.cpu().numpy())

        return Metrics.compute(preds_list, gts_list)

    def fit(self, epochs=10):
        for epoch in range(epochs):

            loss = self.train_one_epoch()
            mae, mse, r2 = self.evaluate(self.val_loader)

            print(f"\nEpoch {epoch+1}")
            print(f"Loss: {loss:.4f}")
            print(f"VAL → MAE: {mae:.2f} | MSE: {mse:.2f} | R2: {r2:.3f}")

            if mae < self.best_mae:
                self.best_mae = mae

                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "mae": mae,
                    "epoch": epoch
                }, self.save_path)

                print(f"Saved model (MAE: {mae:.2f})")

    def test(self):
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, weights_only= False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        mae, mse, r2 = self.evaluate(self.test_loader)

        print("\n===== FINAL TEST =====")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"R2: {r2:.3f}")

# ===== MAIN =====
if __name__ == "__main__":

    TRAIN_DIR = Data_Path.TRAIN_DIR
    VAL_DIR   = Data_Path.VAL_DIR
    TEST_DIR  = Data_Path.TEST_DIR

    GT_TRAIN_DIR = Data_Path.GT_TRAIN_DIR
    GT_VAL_DIR   = Data_Path.GT_VAL_DIR
    GT_TEST_DIR  = Data_Path.GT_TEST_DIR

    CSV_PATH = Data_Path.CSV_PATH

    gt_gen = GTGenerator(CSV_PATH)
    gt_gen.create_gt(TRAIN_DIR, GT_TRAIN_DIR)
    gt_gen.create_gt(VAL_DIR, GT_VAL_DIR)

    print("DONE CREATE GT")

    if not os.path.exists(GT_TEST_DIR):
        gt_gen.create_gt(TEST_DIR, GT_TEST_DIR)

    trainer = Trainer(
        TRAIN_DIR, VAL_DIR, TEST_DIR,
        GT_TRAIN_DIR, GT_VAL_DIR, GT_TEST_DIR
    )

    trainer.fit(epochs=20)
    trainer.test()