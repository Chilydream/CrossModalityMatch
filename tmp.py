import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.Meter import Meter
from model.SyncNetModelFBank import SyncNetModel
from model.AffineModel import AffineModel

a = AffineModel()
b = SyncNetModel()
p = [{'params':a.parameters()}, {'params':b.parameters()}]
o = optim.Adam(p)
