import argparse
import os
from glob import glob

from config.TrainConfig import TRAIN_PARAMETER as TP
from criterion.CrossEntropy import CrossEntropy
from criterion.InfoNCE import InfoNCE
from pytorch_metric_learning.losses import NTXentLoss


def get_console_args():
	parser = argparse.ArgumentParser(description='MNIST GAN')
	parser.add_argument('-m', '--mode', type=str, default=TP['mode'])
	parser.add_argument('-p', '--pretrain_model', type=str, default=TP['pretrain_model'])
	parser.add_argument('--epoch', type=int, default=TP['epoch'])
	parser.add_argument('-b', '--batch_size', type=int, default=TP['batch_size'])
	parser.add_argument('--embedding_size', type=int, default=TP['embedding_size'])
	parser.add_argument('--num_workers', type=int, default=TP['num_workers'])
	parser.add_argument('--lr', type=float, default=TP['lr'])
	parser.add_argument('--temperature', type=float, default=TP['temperature'])
	parser.add_argument('-t', '--temporal_gap', type=int, default=TP['temporal_gap'])
	parser.add_argument('-w', '--merge_win', type=int, default=TP['merge_win'])
	parser.add_argument('-v', '--valid_step', type=int, default=TP['valid_step'])
	parser.add_argument('-d', '--dis_step', type=int, default=TP['dis_step'])
	parser.add_argument('--cpu', action='store_true', default=not TP['gpu'])

	parser.add_argument('-i', '--alphaI', type=float, default=TP['alphaI'])
	parser.add_argument('-c', '--alphaC', type=float, default=TP['alphaC'])

	parser.add_argument('--exp_dir', type=str, default=TP['exp_dir'])
	parser.add_argument('-e', '--exp_num', type=str, default=TP['exp_num'])
	parser.add_argument('--train_list', type=str, default=TP['train_list'])
	parser.add_argument('--test_list', type=str, default=TP['test_list'])

	parser.add_argument('--reduce_method', type=str, default=TP['reduce_method'])
	parser.add_argument('--criterion', type=str)
	parser.add_argument('-a', '--affine', action='store_true', default=TP['affine'])
	parser.add_argument('--affine_lr', type=float)
	args = parser.parse_args()

	TP['mode'] = args.mode
	TP['epoch'] = args.epoch
	TP['batch_size'] = args.batch_size
	TP['embedding_size'] = args.embedding_size
	TP['num_workers'] = args.num_workers
	TP['lr'] = args.lr
	TP['temperature'] = args.temperature
	TP['temporal_gap'] = args.temporal_gap
	TP['merge_win'] = args.merge_win
	TP['valid_step'] = args.valid_step
	TP['dis_step'] = args.dis_step
	TP['gpu'] = not args.cpu

	TP['alphaI'] = args.alphaI
	TP['alphaC'] = args.alphaC

	TP['exp_dir'] = args.exp_dir
	TP['exp_num'] = args.exp_num
	TP['train_list'] = args.train_list
	TP['test_list'] = args.test_list
	TP['reduce_method'] = args.reduce_method
	TP['affine'] = args.affine
	if args.affine_lr is None:
		args.affine_lr = TP['lr']

	if TP['mode'].lower() in {'test', 'eval', 'valid', 'continue'}:
		if os.path.exists(args.pretrain_model):
			TP['pretrain_model'] = args.pretrain_model
		else:
			model_dir = os.path.join(TP['exp_dir'], TP['exp_num'], 'model0*.model')
			model_list = glob(model_dir)
			model_list.sort()
			TP['pretrain_model'] = model_list[-1]

	if args.criterion is not None:
		c_dict = {'InfoNCE': InfoNCE,
		          'NTXentLoss': NTXentLoss,
		          'CrossEntropy': CrossEntropy}
		TP['criterion'] = c_dict[args.criterion]
