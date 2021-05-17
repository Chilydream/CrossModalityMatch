from criterion.InfoNCE import InfoNCE

TRAIN_PARAMETER = {
	'mode': 'train',
	'pretrain_model': '/GPUFS/ruc_tliu_1/fsx/CrossModalityMatch/exp/1/cache/model000000009.model',

	'epoch': 100,
	'batch_size': 30,
	'embedding_size': 1024,
	'num_workers': 4,
	'lr': 1e-5,
	'temperature': 0.07,
	'temporal_gap': 1,
	'merge_win': 3,
	'valid_step': 5,
	'gpu': True,

	'alphaI': 1.0,
	'alphaC': 1.0,

	'exp_dir': 'exp',
	'exp_num': '1',
	'train_list': "data/train.txt",
	'test_list': 'data/test.txt',

	'reduce_method': 'random',
	'criterion': InfoNCE,
	'affine': False,
	'affine_lr': 1e-5,
}
