import torch
import torch.nn as nn

from criterion.CrossEntropy import CrossEntropy
from model.AffineModel import AffineModel

import wandb
import math
import random

wandb.init(project="test-drive", config={
    "learning_rate": 0.001,
    "dropout": 0.2,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
})
config = wandb.config

# Simulating a training or evaluation loop
for x in range(50):
    acc = math.log(1 + x + random.random() * config.learning_rate) + random.random()
    loss = 10 - math.log(1 + x + random.random() + config.learning_rate * x) + random.random()
    # 2️⃣ Log metrics from your script to W&B
    wandb.log({"acc":acc, "loss":loss})

wandb.finish()