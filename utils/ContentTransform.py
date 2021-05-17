import torch
import torch.nn as nn
import torch.nn.functional as F


def ContentTransform(feature_a, feature_b, merge_win=3, gpu=True):
	batch_size, feature_size, time_size = feature_a.size()

	merge_size = (time_size-1)//merge_win+1
	for i_batch in range(batch_size):
		f_a = feature_a[[i_batch], :, :].transpose(2, 0)
		f_b = feature_b[[i_batch], :, :].transpose(2, 0)
		f_a_merge = torch.zeros((merge_size, feature_size, 1), dtype=torch.float32)
		f_b_merge = torch.zeros((merge_size, feature_size, 1), dtype=torch.float32)
		if gpu:
			f_a_merge = f_a_merge.cuda()
			f_b_merge = f_b_merge.cuda()
		for i_time in range(time_size):
			f_a_merge[(i_time-1)//merge_win] += f_a[i_time]
			f_b_merge[(i_time-1)//merge_win] += f_b[i_time]
		yield f_a_merge, f_b_merge
