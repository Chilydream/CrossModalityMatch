import torch
import torch.nn as nn
import torch.nn.functional as F

from model.AffineModel import AffineModel


class UnsupervisedLoss(nn.Module):
	def __init__(self, gpu=True, affine=True):
		super().__init__()
		self.gpu = gpu
		if affine:
			self.affine_net = AffineModel()
			if gpu:
				self.affine_net = self.affine_net.cuda()

	def forward(self, label_tensor, feature_a, feature_b):
		if len(feature_a.size()) == 2:
			feature_a = feature_a.unsqueeze(2)
			feature_b = feature_b.unsqueeze(2)
		elif len(feature_a.size()) != 3 or len(feature_b.size()) != 3:
			raise RuntimeError

		batch_size = feature_a.size()[0]
		label_tensor = torch.arange(batch_size)
		if self.gpu:
			label_tensor = label_tensor.cuda()
		cos_score = F.cosine_similarity(feature_a.expand(-1, -1, batch_size),
		                                feature_b.expand(-1, -1, batch_size).transpose(0, 2))
		cos_score = self.affine_net(cos_score)
		loss = F.cross_entropy(cos_score, label_tensor)
		return loss, cos_score
