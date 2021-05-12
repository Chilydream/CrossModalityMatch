import torch
import torch.nn as nn
import torchsnooper


# @torchsnooper.snoop()
class AffineModel(nn.Module):
	def __init__(self):
		super(AffineModel, self).__init__()
		self.w = nn.Parameter(torch.tensor(10, dtype=torch.float32))
		self.b = nn.Parameter(torch.tensor(-5, dtype=torch.float32))

	def forward(self, x):
		x = x*self.w + self.b
		return x
