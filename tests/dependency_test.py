import torch
import transformers
print(torch.__version__)
print(transformers.__version__)
print("GPU available:", torch.cuda.is_available())
