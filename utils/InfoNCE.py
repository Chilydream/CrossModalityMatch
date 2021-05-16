import torch
import torch.nn as nn
import torch.nn.functional as F

# feature_a 应该为 （batch_size, feature_size, 1)
#             或者 （batch_size, feature_size)
# 返回 InfoNCE loss与 cosine相似度矩阵（该矩阵为 batch_size*batch_size）
from model.AffineModel import AffineModel


class InfoNCE(nn.Module):
	def __init__(self, temperature=0.07, affine=True, **kwargs):
		super().__init__()
		self.temperature = temperature
		self.affine = affine
		if self.affine:
			self.affine_net = AffineModel()
		self.gpu = False
		self.cuda_kwargs = dict()

	def cuda(self, **kwargs):
		network = super(InfoNCE, self).cuda(**kwargs)
		self.gpu = True
		self.cuda_kwargs = kwargs
		return network

	def cpu(self):
		network = super(InfoNCE, self).cpu()
		self.gpu = False
		self.cuda_kwargs = dict()
		return network

	def forward(self, label_tensor, feature_a, feature_b=None):
		# ======================参数调整==================
		cross_modality = True
		if feature_b is None:
			feature_b = feature_a
			cross_modality = False
		if len(feature_a.size()) == 2:
			feature_a = feature_a.unsqueeze(2)
			feature_b = feature_b.unsqueeze(2)
		elif len(feature_a.size()) != 3 or len(feature_b.size()) != 3:
			raise RuntimeError
		batch_size = feature_a.size()[0]
		loss = torch.zeros(1, dtype=torch.float32)
		if self.gpu:
			loss = loss.cuda(**self.cuda_kwargs)

		# =======================计算cosine相似度======================
		cos_score = F.cosine_similarity(feature_a.expand(-1, -1, batch_size),
		                                feature_b.expand(-1, -1, batch_size).transpose(0, 2))
		if self.affine:
			cos_score = self.affine_net(cos_score)

		# =======================计算每个样本的 NTXentloss====================
		for i in range(batch_size):
			numerator = torch.zeros(1, dtype=torch.float32)
			denominator = torch.zeros(1, dtype=torch.float32)
			if self.gpu:
				numerator = numerator.cuda(**self.cuda_kwargs)
				denominator = denominator.cuda(**self.cuda_kwargs)
			for j in range(batch_size):
				if i == j and not cross_modality: continue
				# 如果同模态时，也不跳过 ii对，会如何？
				if label_tensor[i] == label_tensor[j]:
					numerator += torch.exp(cos_score[i, j]/self.temperature)
				denominator += torch.exp(cos_score[i, j]/self.temperature)
			if numerator == 0:
				numerator += 0.01
			loss -= torch.log((numerator/denominator))
		loss /= batch_size
		return loss, cos_score
