import torch
import torch.nn as nn
import torch.nn.functional as F

# feature_a 应该为 （batch_size, feature_size, 1)
#             或者 （batch_size, feature_size)
# 返回 InfoNCE loss与 cosine相似度矩阵（该矩阵为 batch_size*batch_size）
from model.AffineModel import AffineModel


class UniformLoss(nn.Module):
	def __init__(self, temperature=0.07, affine=False, **kwargs):
		super().__init__()
		self.temperature = temperature
		self.affine = affine
		if self.affine:
			self.affine_net = AffineModel()

	def __str__(self):
		return 'UniformLoss'

	def forward(self, label_tensor, feature_a, feature_b=None):
		# ======================参数调整==================
		if feature_b is None:
			feature_b = feature_a
			cross_modality = False
		else:
			cross_modality = True
		if len(feature_a.size()) == 2:
			feature_a = feature_a.unsqueeze(2)
			feature_b = feature_b.unsqueeze(2)
		elif len(feature_a.size()) != 3 or len(feature_b.size()) != 3:
			raise RuntimeError
		batch_size = feature_a.size()[0]
		loss = 0

		# =======================计算cosine相似度======================
		cos_score = F.cosine_similarity(feature_a.expand(-1, -1, batch_size),
		                                feature_b.expand(-1, -1, batch_size).transpose(0, 2))
		if self.affine:
			cos_score = self.affine_net(cos_score)

		cos_prob = F.softmax(cos_score, dim=1)

		# ========================计算每个样本的UniformLoss==============
		for i in range(batch_size):
			for j in range(batch_size):
				if label_tensor[i] == label_tensor[j]:
					loss -= torch.log(cos_prob[i][j])
		loss /= batch_size**2
		return loss, cos_score
