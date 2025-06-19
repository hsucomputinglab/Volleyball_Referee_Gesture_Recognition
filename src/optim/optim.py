import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from src.core.yaml_utils import register


__all__ = ['AdamW', 'SGD', 'Adam', 'MultiStepLR', 'CosineAnnealingLR', 'LambdaLR']



SGD = register(optim.SGD)
Adam = register(optim.Adam)
AdamW = register(optim.AdamW)


MultiStepLR = register(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register(lr_scheduler.CosineAnnealingLR)
LambdaLR = register(lr_scheduler.LambdaLR)
