import torch
import torch.nn as nn

from criterion.CrossEntropy import CrossEntropy
from model.AffineModel import AffineModel


class TestNet(nn.Module):
	def __init__(self, temperature=0.07, affine=True):
		super().__init__()
		self.model = nn.BatchNorm2d(1)

	def forward(self, x):
		x = self.model(x)
		return x


b = TestNet()
print(b.model.__dict__)
