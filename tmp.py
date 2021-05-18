# import wandb
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb

from criterion.CrossEntropy import CrossEntropy
from criterion.InfoNCE import InfoNCE
from model.AffineModel import AffineModel

a = torch.rand((30, 1024, 17), dtype=torch.float32)
time_size = a.size()[2]
b = np.random.randint(0, time_size, 3)
c = a[:, :, b]
print(c.size())
c = c.mean(dim=2)
print(c.size())

# wandb.init(project="test-drive", config={
# 	"learning_rate": 0.001,
# 	"dropout": 0.2,
# 	"architecture": "CNN",
# 	"dataset": "CIFAR-100",
# })
# config = wandb.config
# print(config)
#
# # Simulating a training or evaluation loop
# for x in range(50):
# 	acc = math.log(1+x+random.random()*config.learning_rate)+random.random()
# 	loss = 10-math.log(1+x+random.random()+config.learning_rate*x)+random.random()
# 	wandb.log({"acc": acc, "loss": loss})
#
# wandb.finish()
