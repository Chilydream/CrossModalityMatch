import torch
import torch.nn as nn
import torch.nn.functional as F

from model.AffineModel import AffineModel


class CrossEntropy(nn.Module):
	def __init__(self, affine=False, **kwargs):
		super(CrossEntropy, self).__init__()
		self.affine = affine
		if affine:
			self.affine_net = AffineModel()

	def forward(self, label_tensor, feature_a, feature_b):
		# ========================参数设置======================
		if len(feature_a.size()) == 2:
			feature_a = feature_a.unsqueeze(2)
			feature_b = feature_b.unsqueeze(2)
		elif len(feature_a.size()) != 3 or len(feature_b.size()) != 3:
			raise RuntimeError
		batch_size = feature_a.size()[0]
		loss = 0

		# ========================计算cosine相似度===========================
		cos_score = F.cosine_similarity(feature_a.expand(-1, -1, batch_size),
		                                feature_b.expand(-1, -1, batch_size).transpose(0, 2))
		if self.affine:
			cos_score = self.affine_net(cos_score)
		cos_prob = F.softmax(cos_score, dim=1)

		# ========================计算每个样本的CrossEntropyLoss==============
		for i in range(batch_size):
			true_label_cnt = 0
			tmp_loss = 0
			for j in range(batch_size):
				if label_tensor[i]==label_tensor[j]:
					tmp_loss -= torch.log(cos_prob[i][j])
					true_label_cnt += 1
			tmp_loss = tmp_loss/true_label_cnt
			loss += tmp_loss

		loss /= batch_size
		return loss, cos_score
