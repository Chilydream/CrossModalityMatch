import torch
import numpy as np


def SampleFromTime(a, method='random'):
	if method.lower() in {'random', 'r'}:
		batch_size = a.size()[0]
		time_size = a.size()[2]
		b = []
		for i in range(batch_size):
			b.append(a[i, :, np.random.randint(0, time_size)])
		c = torch.stack(b)
		return c
	elif method.lower() in {'mean', 'm'}:
		b = a.mean(dim=2)
		return b
