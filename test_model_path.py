import os

model_path = "./Musetalk/models/dwpose/dw-ll_ucoco_384.pth"

print("Absolute path:", os.path.abspath(model_path))
print("Exists:", os.path.exists(model_path))
print("Readable:", os.access(model_path, os.R_OK))
