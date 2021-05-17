import os
import time
import torch
from torch import nn

from model.SyncNetModelFBank import SyncNetModel
from utils.DatasetLoader import MyDataLoader
from utils.GetConsoleArgs import get_console_args
from utils.Meter import Meter
from utils.accuracy import topk_acc, rand_acc
from utils.sampler import SampleFromTime
from config.TrainConfig import TRAIN_PARAMETER as TP


# @torchsnooper.snoop()
def acc_valid(valid_loader, sync_net, criterion, logfile=None):
	print('\nStart evaluate')
	valid_timer = Meter('Valid Timer', 'time', ':3.0f')
	valid_id_acc = Meter('Valid ID ACC', 'avg', ':.2f', '%,')
	valid_id_rand_acc = Meter('Valid ID Rand ACC', 'avg', ':.2f', '%')
	valid_loss = Meter('Valid Loss', 'avg', ':.2f')
	valid_timer.set_start_time(time.time())
	with torch.no_grad():
		for data in valid_loader:
			data_video, data_audio, data_label = data
			data_video, data_audio = data_video.cuda(), data_audio.cuda()
			_, audio_id = sync_net.forward_aud(data_audio)
			_, video_id = sync_net.forward_vid(data_video)
			# (batch_size, feature, time_size)

			audio_random_id = SampleFromTime(audio_id).unsqueeze(2)
			video_random_id = SampleFromTime(video_id).unsqueeze(2)
			# (batch_size, feature, 1)

			id_loss, id_score = criterion(data_label, audio_random_id, video_random_id)
			id_acc = topk_acc(id_score, data_label)
			id_rand_acc = rand_acc(data_label)

			batch_loss = id_loss

			valid_timer.update(time.time())
			valid_id_acc.update(id_acc*100)
			valid_id_rand_acc.update(id_rand_acc*100)
			valid_loss.update(batch_loss.item())

			print('\r', valid_timer, valid_loss, valid_id_acc, valid_id_rand_acc, end='       ')
			torch.cuda.empty_cache()
		if logfile is not None:
			print(valid_loss, valid_id_acc, file=logfile)
	print('')


def main():
	# ===========================参数设定===============================
	get_console_args()
	start_epoch = 0
	batch_size = TP['batch_size']
	dumb_label = torch.arange(batch_size)
	if TP['gpu']:
		dumb_label = dumb_label.cuda()
	learning_rate = TP['lr']
	cur_exp_path = os.path.join(TP['exp_dir'], TP['exp_num'])
	cache_dir = os.path.join(cur_exp_path, 'cache')
	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)
	path_train_log = os.path.join(cur_exp_path, 'train.log')
	file_train_log = open(path_train_log, 'w')

	# ============================模型载入===============================
	print('Start loading model')
	sync_net = SyncNetModel(nOut=TP['embedding_size'], stride=TP['temporal_gap'])
	criterion = TP['criterion'](temperature=TP['temperature'], affine=TP['affine'])
	TP['affine'] = TP['affine'] and 'affine_net' in criterion.__dir__()
	p_list = [{'params': sync_net.parameters(), 'lr': TP['lr']}]
	if TP['affine']:
		affine_net = criterion.affine_net
		p_list.append({'params': affine_net.parameters(), 'lr': TP['affine_lr']})
	if TP['gpu']:
		sync_net = sync_net.cuda()
		criterion = criterion.cuda()
	optimizer = torch.optim.Adam(p_list, learning_rate)
	sync_net.train()
	criterion.train()

	# ============================度量载入===============================
	epoch_loss_final = Meter('Final Loss', 'avg', ':.2f')
	epoch_id_loss = Meter('ID Loss', 'avg', ':.2f')
	epoch_id_acc = Meter('ID ACC', 'avg', ':.2f', '%,')
	epoch_id_rand_acc = Meter('ID Rand ACC', 'avg', ':.2f', '%,')
	epoch_timer = Meter('Time', 'time', ':3.0f')
	epoch_reset_list = [epoch_loss_final, epoch_id_loss, epoch_id_acc, epoch_timer,
	                    epoch_id_rand_acc]
	print('%sTrain Parameters%s'%('='*20, '='*20))
	print(TP)
	print('')
	print(TP, file=file_train_log)

	# ============================数据载入===============================
	loader_timer = Meter('Time', 'time', ':3.0f', end='')
	print('Start loading dataset')
	loader_timer.set_start_time(time.time())
	if TP['mode'].lower() in ['train', 'continue']:
		train_loader = MyDataLoader(TP['train_list'], batch_size, TP['num_workers'])
	valid_loader = MyDataLoader(TP['test_list'], batch_size, TP['num_workers'])
	loader_timer.update(time.time())
	print('Finish loading dataset', loader_timer)

	# ========================预加载模型================================
	if TP['mode'].lower() in ['test', 'valid', 'eval']:
		model_ckpt = torch.load(TP['pretrain_model'])
		sync_net.load_state_dict(model_ckpt['sync_net'])
		if 'affine' in model_ckpt.keys() and TP['affine']:
			affine_net.load_state_dict(model_ckpt['affine_net'])
		acc_valid(valid_loader, sync_net, criterion)
		return
	elif TP['mode'].lower() in ['continue']:
		print('Loading pretrained model...')
		model_ckpt = torch.load(TP['pretrain_model'])
		sync_net.load_state_dict(model_ckpt['sync_net'])
		if 'affine' in model_ckpt.keys() and TP['affine']:
			affine_net.load_state_dict(model_ckpt['affine_net'])
		optimizer.load_state_dict(model_ckpt['optimizer'])
		start_epoch = model_ckpt['epoch']

	# ============================开始训练===============================
	print('Start Training')
	for epoch in range(start_epoch, TP['epoch']):
		print('\nEpoch: %d'%epoch)
		batch_cnt = 0
		epoch_timer.set_start_time(time.time())
		for data in train_loader:
			data_video, data_audio, data_label = data
			if TP['gpu']:
				data_video, data_audio = data_video.cuda(), data_audio.cuda()
			_, audio_id = sync_net.forward_aud(data_audio)
			_, video_id = sync_net.forward_vid(data_video)
			# (batch_size, feature, time_size)

			audio_random_id = SampleFromTime(audio_id, TP['reduce_method'])
			video_random_id = SampleFromTime(video_id, TP['reduce_method'])
			# (batch_size, feature)

			batch_id_loss, id_score = criterion(dumb_label, audio_random_id, video_random_id)
			# batch_id_loss, id_score = criterion(data_label, audio_random_id, video_random_id)
			batch_id_acc = topk_acc(id_score, data_label, 1)
			id_rand_acc = rand_acc(data_label)

			batch_loss_final = batch_id_loss
			batch_loss_final.backward()
			nn.utils.clip_grad_norm_(sync_net.parameters(), max_norm=10, norm_type=2)
			if TP['affine']:
				nn.utils.clip_grad_norm_(affine_net.parameters(), max_norm=10, norm_type=2)
			optimizer.step()

			# ==========计量更新============================
			epoch_id_acc.update(batch_id_acc*100)
			epoch_id_rand_acc.update(id_rand_acc*100)
			epoch_id_loss.update(batch_id_loss.item())
			epoch_loss_final.update(batch_loss_final.item())
			epoch_timer.update(time.time())
			batch_cnt += 1
			print('\rBatch:(%02d/%d)'%(batch_cnt, len(train_loader)),
			      epoch_timer, epoch_id_acc, epoch_id_rand_acc,
			      epoch_loss_final, epoch_id_loss, end='             ')
		print('Epoch:', epoch, epoch_loss_final, epoch_id_acc, epoch_id_rand_acc,
		      file=file_train_log)
		for meter in epoch_reset_list:
			meter.reset()
		if TP['gpu']:
			torch.cuda.empty_cache()
		# torch.save(sync_net.state_dict(), cache_dir+"/model%09d.model"%epoch)
		if TP['affine']:
			torch.save({'epoch': epoch+1,
			            'sync_net': sync_net.state_dict(),
			            'affine_net': affine_net.state_dict(),
			            'optimizer': optimizer.state_dict()},
			           cache_dir+"/model%09d.model"%epoch)
		else:
			torch.save({'epoch': epoch+1,
			            'sync_net': sync_net.state_dict(),
			            'optimizer': optimizer.state_dict()},
			           cache_dir+"/model%09d.model"%epoch)
		if (epoch+1)%TP['valid_step'] == 0:
			sync_net.eval()
			criterion.eval()
			acc_valid(valid_loader, sync_net, criterion, file_train_log)
			sync_net.train()
			criterion.train()
	file_train_log.close()


if __name__ == '__main__':
	main()
