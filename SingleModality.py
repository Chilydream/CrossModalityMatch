import os
import time
import torch
from pytorch_metric_learning import losses

from model.SyncNetModelFBank import SyncNetModel
from utils.DatasetLoader import MyDataLoader
from criterion.InfoNCE import InfoNCE
from utils.Meter import Meter
from utils.accuracy import topk_acc
from utils.sampler import SampleFromTime


# @torchsnooper.snoop()
def acc_valid(valid_loader, model, temperature=0.07):
	print('\nStart evaluate')
	valid_timer = Meter('Valid Timer', 'time', ':3.0f')
	valid_timer.set_start_time(time.time())
	for data in valid_loader:
		data_video, data_audio, data_label = data
		data_video, data_audio = data_video.cuda(), data_audio.cuda()
		_, audio_id = model.forward_aud(data_audio)
		_, video_id = model.forward_vid(data_video)
		# (batch_size, feature, time_size)

		audio_random_id = SampleFromTime(audio_id).unsqueeze(2)
		video_random_id = SampleFromTime(video_id).unsqueeze(2)
		# (batch_size, feature, 1)

		audio_id_loss, audio_score = InfoNCE(data_label, audio_random_id, temperature=temperature)
		video_id_loss, video_score = InfoNCE(data_label, video_random_id, temperature=temperature)
		audio_acc = topk_acc(audio_score, data_label)
		video_acc = topk_acc(video_score, data_label)
		valid_loss = audio_id_loss+video_id_loss
		valid_timer.update(time.time())
		print('\r%s Valid Loss: %.2f, Audio ACC: %.2f, Video ACC: %.2f   '%
		      (valid_timer, valid_loss.item(), audio_acc*100, video_acc*100), end='')
		torch.cuda.empty_cache()


def main():
	# =================参数设置=================================
	batch_size = 30
	learning_rate = 0.001
	exp_path = 'data/exp12/'
	model_path = exp_path+'model'
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	result_path = exp_path+'result.txt'
	fw = open(result_path, 'w')

	# =================加载数据================================
	loader_timer = Meter('Time', 'time', ':3.0f')
	print('Start loading dataset')
	loader_timer.set_start_time(time.time())
	# train_loader = MyDataLoader("data/test.txt", batch_size)
	train_loader = MyDataLoader("data/train.txt", batch_size)
	valid_loader = MyDataLoader("data/test.txt", batch_size)
	loader_timer.update(time.time())
	print('Finish loading dataset', loader_timer)

	# ================加载模型==================================
	model = SyncNetModel(stride=5).cuda()
	optim = torch.optim.Adam(model.parameters(), learning_rate)
	criterion = losses.NTXentLoss(0.07)
	temperature = 0.07
	loss = Meter('Loss', 'avg', ':.2f')
	epoch_timer = Meter('Time', 'time', ':3.0f')

	# ================开始训练==================================
	for epoch in range(100):
		print('\nEpoch: %d'%epoch)
		batch_cnt = 0
		epoch_timer.set_start_time(time.time())
		for data in train_loader:
			data_video, data_audio, data_label = data
			data_video, data_audio = data_video.cuda(), data_audio.cuda()
			_, audio_id = model.forward_aud(data_audio)
			_, video_id = model.forward_vid(data_video)
			# (batch_size, feature, time_size)

			audio_random_id = SampleFromTime(audio_id)
			video_random_id = SampleFromTime(video_id)
			# (batch_size, feature)

			# audio_id_loss, _ = InfoNCE(data_label, audio_random_id, temperature=temperature)
			# video_id_loss, _ = InfoNCE(data_label, video_random_id, temperature=temperature)
			audio_id_loss = criterion(audio_random_id, data_label)
			video_id_loss = criterion(video_random_id, data_label)
			# audio_acc = topk_acc(audio_score, data_label)
			# video_acc = topk_acc(video_score, data_label)

			final_loss = audio_id_loss+video_id_loss
			final_loss.backward()
			optim.step()
			batch_cnt += 1
			loss.update(final_loss.item())
			epoch_timer.update(time.time())
			print('\rBatch:(%02d/%d)'%(batch_cnt, len(train_loader)),
			      epoch_timer, loss, '            ', end='')
		# print('Loss: %.2f\nAudio ID ACC:%.2f%%\tVideo ID ACC:%.2f%%'%(final_loss.item(), audio_acc*100, video_acc*100))
		print('Epoch: %3d\t%s'%(epoch, loss), file=fw)
		loss.reset()
		torch.cuda.empty_cache()
		torch.save(model.state_dict(), model_path+"/model%09d.model"%epoch)
	fw.close()


main()
