from utils.InfoNCE import InfoNCE
from pytorch_metric_learning.losses import NTXentLoss

TRAIN_PARAMETER = {
	'epoch': 100,
	'batch_size': 30,
	'embedding_size': 1024,
	'num_workers': 4,
	'lr': 1e-3,
	'temperature': 0.07,
	'temporal_gap': 5,
	'valid_step': 5,
	'gpu': True,

	'exp_dir': 'exp',
	'exp_num': '1',
	'train_list': "data/train.txt",
	'test_list': 'data/test.txt',

	'reduce_method': 'mean',
	# 'criterion': NTXentLoss,
	'criterion': InfoNCE,
	'affine': False,
}
