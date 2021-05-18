# import wandb
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from criterion.CrossEntropy import CrossEntropy
from criterion.InfoNCE import InfoNCE
from model.AffineModel import AffineModel

a = torch.rand((30, 16, 1), dtype=torch.float32).cuda()
b = torch.rand((30, 16, 1), dtype=torch.float32).cuda()
# criterion = InfoNCE(affine=True)
criterion = CrossEntropy(affine=True)
optimizer = torch.optim.Adam(criterion.parameters(), lr=0.001)
label = torch.arange(30,).cuda()
cos_score1 = F.cosine_similarity(a.expand(-1, -1, 30), b.expand(-1, -1, 30).transpose(0, 2))
loss1 = F.cross_entropy(cos_score1, label)
print(loss1)

for i in range(1000):
	loss2, _ = criterion(label, a, b)
	optimizer.zero_grad()
	loss2.backward()
	optimizer.step()
	print(loss2.item())


# wandb.init(project="test-drive", config={
#     "learning_rate": 0.001,
#     "dropout": 0.2,
#     "architecture": "CNN",
#     "dataset": "CIFAR-100",
# })
# config = wandb.config
#
# # Simulating a training or evaluation loop
# for x in range(50):
#     acc = math.log(1 + x + random.random() * config.learning_rate) + random.random()
#     loss = 10 - math.log(1 + x + random.random() + config.learning_rate * x) + random.random()
#     # 2️⃣ Log metrics from your script to W&B
#     wandb.log({"acc":acc, "loss":loss})
#
# wandb.finish()
