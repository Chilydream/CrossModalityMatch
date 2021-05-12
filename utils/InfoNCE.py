import torch
import torch.nn as nn
import torch.nn.functional as F


# feature_a 应该为 （batch_size, feature_size, 1)
#             或者 （batch_size, feature_size)
# 返回 InfoNCE loss与 cosine相似度矩阵（该矩阵为 batch_size*batch_size）
def InfoNCE(label_tensor, feature_a, feature_b=None, temperature=0.07):
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
	cos_score = F.cosine_similarity(feature_a.expand(-1, -1, batch_size),
	                                feature_b.expand(-1, -1, batch_size).transpose(0, 2))
	loss = torch.zeros(1, dtype=torch.float32).cuda()
	for i in range(batch_size):
		numerator = torch.zeros(1, dtype=torch.float32).cuda()
		denominator = torch.zeros(1, dtype=torch.float32).cuda()
		for j in range(batch_size):
			if i == j and not cross_modality: continue
			# 如果同模态时，也不跳过 ii对，会如何？
			if label_tensor[i] == label_tensor[j]:
				numerator += torch.exp(cos_score[i, j]/temperature)
			denominator += torch.exp(cos_score[i, j]/temperature)
		if numerator == 0:
			numerator += 0.01
		loss -= torch.log((numerator/denominator))
	return loss, cos_score
