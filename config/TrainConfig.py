from criterion.InfoNCE import InfoNCE

TRAIN_PARAMETER = {
	'mode': 'train',
	'pretrain_model': '',

	'epoch': 100,
	'batch_size': 30,
	'embedding_size': 1024,
	'num_workers': 4,
	'lr': 1e-3,
	'temperature': 0.07,
	'temporal_gap': 3,
	'merge_win': 2 ,
	'valid_step': 5,
	'dis_step': 1,
	'gpu': True,

	'alphaI': 1.0,
	'alphaC': 5.0,

	'exp_dir': 'exp',
	'exp_num': '1',
	'train_list': "data/train_part.txt",
	'test_list': 'data/test_part.txt',

	'reduce_method': 'rm',
	'criterion': InfoNCE,
	'affine': False,
	'affine_lr': 1e-5,
}
