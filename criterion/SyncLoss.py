import torch
import torch.nn as nn
import torch.nn.functional as F


class SyncLoss(nn.Module):
	def __init__(self):
		super(SyncLoss, self).__init__()
		self.gpu = False
		self.cuda_kwargs = dict()

	def cuda(self, **kwargs):
		model = super(SyncLoss, self).cuda(**kwargs)
		self.cuda_kwargs = kwargs
		self.gpu = True
		return model

	def cpu(self):
		model = super(SyncLoss, self).cpu()
		self.gpu = False
		self.cuda_kwargs = dict()
		return model

	def forward(self, data_label, feature_a, feature_b, criterion):
		batch_size = data_label.size()[0]
		feature_size = data_label.size()[1]
		time_size = data_label.size()[2]

		merge_size = (time_size-1)//3+1
		loss = 0
		for i_batch in range(batch_size):
			f_a = feature_a[[i_batch], :, :].transpose(2, 0)
			f_b = feature_b[[i_batch], :, :].transpose(2, 0)
			f_a_merge = torch.zeros((merge_size, feature_size, 1), dtype=torch.float32)
			f_b_merge = torch.zeros((merge_size, feature_size, 1), dtype=torch.float32)
			if self.gpu:
				f_a_merge = f_a_merge.cuda(**self.cuda_kwargs)
				f_b_merge = f_b_merge.cuda(**self.cuda_kwargs)
			for i_time in range(time_size):
				f_a_merge[(i_time-1)//3] += f_a[i_time]
				f_b_merge[(i_time-1)//3] += f_b[i_time]
			cos_score = F.cosine_similarity(f_a_merge.expand(-1, -1, merge_size),
			                                f_b_merge.expand(-1, -1, merge_size).transpose(0, 2))
			loss += criterion(cos_score)
