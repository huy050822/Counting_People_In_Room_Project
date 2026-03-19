import torch
from torch.utils.data import DataLoader
from models.data_path import Data_Path
from models.model_cpv import CrowdDataset
from models.model_cpv import CrowdCountingModel
import time
import psutil, os

model = CrowdCountingModel()

checkpoint = torch.load("trained_model.pth", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

dataset = CrowdDataset(
    Data_Path.TEST_DIR,
    Data_Path.GT_TEST_DIR
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

process = psutil.Process(os.getpid())

total_time = 0
total_samples = 0

with torch.no_grad():
    for imgs, _ in dataloader:
        start = time.time()
        _ = model(imgs)
        end = time.time()

        total_time += (end - start)
        total_samples += imgs.size(0)

latency = total_time / total_samples
fps = 1 / latency

memory = process.memory_info().rss / 1024**2  

print(f"Latency: {latency:.4f}s")
print(f"FPS: {fps:.2f}")
print(f"Memory Usage: {memory:.2f} MB")