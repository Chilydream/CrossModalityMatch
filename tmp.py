import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_metric_learning import losses

from utils.CrossEntropy import CrossEntropy
from utils.Meter import Meter
from model.SyncNetModelFBank import SyncNetModel
from model.AffineModel import AffineModel
from utils.sampler import SampleFromTime


class TestNet(nn.Module):
	def __init__(self, temperature=0.07, affine=True):
		super().__init__()
		self.temperature = temperature
		self.affine_net = AffineModel()
		self.gpu = False
		self.cuda_kwargs = None

	def cuda(self, **kwargs):
		testnet = super(TestNet, self).cuda()
		self.gpu = True
		self.cuda_kwargs = kwargs
		return testnet

	def forward(self):
		a = torch.zeros(1)
		if self.gpu:
			a = a.cuda(**self.cuda_kwargs)
		return a


b = losses.NTXentLoss