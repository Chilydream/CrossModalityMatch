import argparse
from config.TrainConfig import TRAIN_PARAMETER as TP
from utils.CrossEntropy import CrossEntropy
from utils.InfoNCE import InfoNCE
from pytorch_metric_learning.losses import NTXentLoss


def get_console_args():
	parser = argparse.ArgumentParser(description='MNIST GAN')
	parser.add_argument('-e', '--epoch', type=int, default=TP['epoch'])
	parser.add_argument('-b', '--batch_size', type=int, default=TP['batch_size'])
	parser.add_argument('--embedding_size', type=int, default=TP['embedding_size'])
	parser.add_argument('--num_workers', type=int, default=TP['num_workers'])
	parser.add_argument('--lr', type=float, default=TP['lr'])
	parser.add_argument('--temperature', type=float, default=TP['temperature'])
	parser.add_argument('--temporal_gap', type=int, default=TP['temporal_gap'])
	parser.add_argument('-v', '--valid_step', type=int, default=TP['valid_step'])
	parser.add_argument('--cpu', action='store_true', default=not TP['gpu'])

	parser.add_argument('--exp_dir', type=str, default=TP['exp_dir'])
	parser.add_argument('--exp_num', type=str, default=TP['exp_num'])
	parser.add_argument('--train_list', type=str, default=TP['train_list'])
	parser.add_argument('--test_list', type=str, default=TP['test_list'])

	parser.add_argument('--reduce_method', type=str, default=TP['reduce_method'])
	parser.add_argument('--criterion', type=str)
	parser.add_argument('--affine', action='store_true', default=TP['affine'])
	args = parser.parse_args()

	TP['epoch'] = args.epoch
	TP['batch_size'] = args.batch_size
	TP['embedding_size'] = args.embedding_size
	TP['num_workers'] = args.num_workers
	TP['lr'] = args.lr
	TP['temperature'] = args.temperature
	TP['temporal_gap'] = args.temporal_gap
	TP['valid_step'] = args.valid_step
	TP['gpu'] = not args.cpu
	TP['exp_dir'] = args.exp_dir
	TP['exp_num'] = args.exp_num
	TP['train_list'] = args.train_list
	TP['test_list'] = args.test_list
	TP['reduce_method'] = args.reduce_method

	if args.criterion is not None:
		c_dict = {'InfoNCE': InfoNCE,
		          'NTXentLoss': NTXentLoss,
		          'CrossEntropy': CrossEntropy}
		TP['criterion'] = c_dict[args.criterion]
