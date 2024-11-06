
import random
import torch
import numpy as np



def set_seed(seed):     # 设置 Python 内置随机数生成器的种子     
    random.seed(seed)      # 设置 NumPy 的随机种子     
    np.random.seed(seed)      # 设置 PyTorch 的随机种子     
    torch.manual_seed(seed)      # 如果你在使用 GPU，还需要设置以下两项
    if torch.cuda.is_available():         
        torch.cuda.manual_seed(seed)         
        torch.cuda.manual_seed_all(seed)  # 如果你使用多个 GPU# 确保结果是可复现的     
        torch.backends.cudnn.deterministic = True     
        torch.backends.cudnn.benchmark = False