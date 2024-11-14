import subprocess
subprocess.run(["src/ml/clamp/midi2abc","data/datasets/20240621-bak/play/20231220-080-01_0000-0005.mid"])
import torch
x=torch.load("data/outputs/20240126-050-04_0335-0345.pt")
print(x.shape)