import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper

from model.AffineModel import AffineModel
from model.SyncNetModelFBank import SyncNetModel
from utils.DatasetLoader import MyDataLoader
from utils.InfoNCE import InfoNCE
from utils.Meter import Meter
from utils.accuracy import topk_acc
from utils.sampler import SampleFromTime


# @torchsnooper.snoop()
def acc_valid(valid_loader, sync_net, temperature=0.07):
	print('\nStart evaluate')
	valid_timer = Meter('Valid Timer', 'time', ':3.0f')
	valid_id_acc = Meter('Valid ID ACC', 'avg', ':.2f', '%,')
	valid_loss = Meter('Valid Loss', 'avg', ':.2f')
	valid_timer.set_start_time(time.time())
	for data in valid_loader:
		data_video, data_audio, data_label = data
		data_video, data_audio = data_video.cuda(), data_audio.cuda()
		_, audio_id = sync_net.forward_aud(data_audio)
		_, video_id = sync_net.forward_vid(data_video)
		# (batch_size, feature, time_size)

		audio_random_id = SampleFromTime(audio_id).unsqueeze(2)
		video_random_id = SampleFromTime(video_id).unsqueeze(2)
		# (batch_size, feature, 1)

		id_loss, id_score = InfoNCE(data_label, audio_random_id, video_random_id,
		                                     temperature=temperature)
		id_acc = topk_acc(id_score, data_label)
		batch_loss = id_loss

		valid_timer.update(time.time())
		valid_id_acc.update(id_acc*100)
		valid_loss.update(batch_loss.item())

		print('\r', valid_timer, valid_loss, valid_id_acc, end='       ')
		torch.cuda.empty_cache()
	print('')


def main():
	# ===========================参数设定===============================
	batch_size = 30
	learning_rate = 0.001
	experiment_path = 'exp/13/'
	cache_path = experiment_path+'cache'
	if not os.path.exists(cache_path):
		os.makedirs(cache_path)
	path_train_log = experiment_path+'train.log'
	file_train_log = open(path_train_log, 'w')

	# ============================数据载入===============================
	loader_timer = Meter('Time', 'time', ':3.0f', end='')
	print('Start loading dataset')
	loader_timer.set_start_time(time.time())
	# train_loader = MyDataLoader("data/test.txt", batch_size)
	train_loader = MyDataLoader("data/train.txt", batch_size)
	valid_loader = MyDataLoader("data/test.txt", batch_size)
	loader_timer.update(time.time())
	print('Finish loading dataset', loader_timer)

	# ============================模型载入===============================
	sync_net = SyncNetModel(stride=5).cuda()
	affine_net = AffineModel()
	p_list = [{'params':sync_net.parameters()}, {'params':affine_net.parameters()}]
	optimizer = torch.optim.Adam(p_list, learning_rate)
	temperature = 0.07
	epoch_loss_final = Meter('Final Loss', 'avg', ':.2f')
	epoch_id_loss = Meter('ID Loss', 'avg', ':.2f')
	epoch_id_acc = Meter('ID ACC', 'avg', ':.2f', '%,')
	epoch_timer = Meter('Time', 'time', ':3.0f')
	epoch_reset_list = [epoch_loss_final, epoch_id_loss, epoch_id_acc, epoch_timer]

	# ============================开始训练===============================
	print('Start Training')
	for epoch in range(100):
		print('\nEpoch: %d'%epoch)
		batch_cnt = 0
		epoch_timer.set_start_time(time.time())
		for data in train_loader:
			data_video, data_audio, data_label = data
			data_video, data_audio = data_video.cuda(), data_audio.cuda()
			_, audio_id = sync_net.forward_aud(data_audio)
			_, video_id = sync_net.forward_vid(data_video)
			# (batch_size, feature, time_size)

			audio_random_id = SampleFromTime(audio_id).unsqueeze(2)
			video_random_id = SampleFromTime(video_id).unsqueeze(2)
			video_random_id = affine_net(video_random_id)
			# (batch_size, feature)

			batch_id_loss, id_score = InfoNCE(data_label, audio_random_id, video_random_id,
			                     temperature=temperature)
			batch_id_acc = topk_acc(id_score, data_label, 1)

			batch_loss_final = batch_id_loss
			batch_loss_final.backward()
			optimizer.step()

			# ==========计量更新============================
			epoch_id_acc.update(batch_id_acc*100)
			epoch_id_loss.update(batch_id_loss.item())
			epoch_loss_final.update(batch_loss_final.item())
			epoch_timer.update(time.time())
			batch_cnt += 1
			print('\rBatch:(%02d/%d)'%(batch_cnt, len(train_loader)),
			      epoch_timer, epoch_id_acc, epoch_loss_final, epoch_id_loss, end='              ')
		print('Epoch: %3d\t%s'%(epoch, epoch_loss_final), file=file_train_log)
		for meter in epoch_reset_list:
			meter.reset()
		torch.cuda.empty_cache()
		torch.save(sync_net.state_dict(), cache_path+"/model%09d.model"%epoch)
		if (epoch+1)%5==0:
			acc_valid(valid_loader, sync_net)
	file_train_log.close()


if __name__ == '__main__':
	main()
