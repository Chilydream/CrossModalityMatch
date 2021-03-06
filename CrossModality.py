import math
import os
import time
import torch
from torch import nn
import torch.nn.functional as F
import wandb

from criterion.UniformLoss import UniformLoss
from model.SyncNetModelFBank import SyncNetModel
from utils.ContentTransform import ContentTransform
from utils.DatasetLoader import MyDataLoader
from utils.GetConsoleArgs import get_console_args
from utils.Meter import Meter
from utils.accuracy import topk_acc, rand_acc
from utils.sampler import SampleFromTime
from config.TrainConfig import TRAIN_PARAMETER as TP


# @torchsnooper.snoop()
def acc_valid(valid_loader, sync_net, criterion, logfile=None):
	print('\nStart evaluate')
	batch_size = TP['batch_size']
	time_size = math.ceil(40/TP['temporal_gap'])
	merge_win = TP['merge_win']
	merge_size = (time_size-1)//merge_win+1
	alphaI = TP['alphaI']
	alphaC = TP['alphaC']
	dumb_id_label = torch.arange(batch_size)
	dumb_ct_label = torch.arange(merge_size)

	valid_timer = Meter('Valid Timer', 'time', ':3.0f')
	valid_id_acc = Meter('ID ACC', 'avg', ':.2f', '%,')
	valid_id_rand_acc = Meter('ID Rand ACC', 'avg', ':.2f', '%')
	valid_ct_acc = Meter('CT ACC', 'avg', ':.2f', '%,')
	valid_ct_rand_acc = Meter('CT Rand ACC', 'avg', ':.2f', '%,')
	valid_ct_rand_acc.update(100/merge_size)
	valid_loss = Meter('Loss', 'avg', ':.2f')
	valid_id_loss = Meter('ID Loss', 'avg', ':.2f')
	valid_ct_loss = Meter('CT Loss', 'avg', ':.2f')
	valid_timer.set_start_time(time.time())
	with torch.no_grad():
		for data in valid_loader:
			data_video, data_audio, data_label = data
			data_video, data_audio = data_video.cuda(), data_audio.cuda()
			audio_ct, audio_id = sync_net.forward_aud(data_audio)
			video_ct, video_id = sync_net.forward_vid(data_video)
			# (batch_size, feature, time_size)

			# =========================计算身份损失================================
			audio_random_id = SampleFromTime(audio_id, TP['reduce_method']).unsqueeze(2)
			video_random_id = SampleFromTime(video_id, TP['reduce_method']).unsqueeze(2)
			# (batch_size, feature, 1)

			id_loss, id_score = criterion(dumb_id_label, audio_random_id, video_random_id)
			id_acc = topk_acc(id_score, data_label)
			id_rand_acc = rand_acc(data_label)

			# =========================计算内容损失================================
			ct_loss = 0
			for a_merge, v_merge in ContentTransform(audio_ct, video_ct, merge_win, TP['gpu']):
				tmp_ct_loss, tmp_ct_score = criterion(dumb_ct_label, a_merge, v_merge)
				ct_loss += tmp_ct_loss
				valid_ct_acc.update(topk_acc(tmp_ct_score, dumb_ct_label)*100)
			ct_loss /= batch_size

			batch_loss = alphaI*id_loss+alphaC*ct_loss

			valid_timer.update(time.time())
			valid_id_acc.update(id_acc*100)
			valid_id_rand_acc.update(id_rand_acc*100)
			valid_id_loss.update(id_loss.item())
			valid_ct_loss.update(ct_loss.item())
			valid_loss.update(batch_loss.item())

			print('\r', valid_timer, valid_loss,
			      valid_id_acc, valid_id_rand_acc, valid_id_loss,
			      valid_ct_acc, valid_ct_rand_acc, valid_ct_loss,
			      end='    ')
			torch.cuda.empty_cache()
		if valid_loader.batch_size==2:
			valid_log = {'valid 2id acc': valid_id_acc.avg}
		else:
			valid_log = {'valid loss': valid_loss.avg,
			             'valid id loss': valid_id_loss.avg,
			             'valid ct loss': valid_ct_loss.avg,
			             'valid id acc': valid_id_acc.avg,
			             'valid ct acc': valid_ct_acc.avg}
		if logfile is not None:
			print(valid_loss, valid_id_acc, valid_id_rand_acc, valid_id_loss,
			      valid_ct_acc, valid_ct_rand_acc, valid_ct_loss,
			      file=logfile)
	print('')
	return valid_log


def dis_info(loader, sync_net, optimizer):
	batch_size = TP['batch_size']
	time_size = math.ceil(40/TP['temporal_gap'])
	merge_win = TP['merge_win']
	merge_size = (time_size-1)//merge_win+1
	dumb_id_label = torch.arange(batch_size)
	dumb_ct_label = torch.arange(merge_size)
	criterion = UniformLoss()
	epoch_ct2id_loss = Meter('CT2ID Loss', 'avg', ':.2f')
	epoch_ct2id_acc = Meter('CT2ID ACC', 'avg', ':.2f', '%')
	epoch_id2ct_loss = Meter('ID2CT Loss', 'avg', ':.2f')
	epoch_id2ct_acc = Meter('ID2CT ACC', 'avg', ':.2f', '%')
	epoch_timer = Meter('Disentanglement Time', 'time', ':.0f')
	epoch_timer.set_start_time(time.time())

	for data in loader:
		data_video, data_audio, data_label = data
		if TP['gpu']:
			data_video, data_audio = data_video.cuda(), data_audio.cuda()
		audio_ct, audio_id = sync_net.forward_aud(data_audio)
		video_ct, video_id = sync_net.forward_vid(data_video)

		audio_random_ct = SampleFromTime(audio_ct, TP['reduce_method'])
		# video_random_ct = SampleFromTime(video_ct, TP['reduce_method'])
		# audio_random_id = SampleFromTime(audio_id, TP['reduce_method'])
		video_random_id = SampleFromTime(video_id, TP['reduce_method'])

		batch_ct2id_loss, ct2id_score = criterion(dumb_id_label, audio_random_ct, video_random_id)
		batch_ct2id_acc = topk_acc(ct2id_score, data_label, 1)

		batch_id2ct_loss = 0
		for a_merge_id, v_merge_ct in ContentTransform(audio_id, video_ct):
			tmp_loss, tmp_score = criterion(dumb_ct_label, a_merge_id, v_merge_ct)
			batch_id2ct_loss += tmp_loss
			epoch_id2ct_acc.update(topk_acc(tmp_score, dumb_ct_label)*100)

		optimizer.zero_grad()
		final_loss = batch_id2ct_loss+batch_ct2id_loss
		final_loss.backward()
		optimizer.step()

		epoch_ct2id_loss.update(batch_ct2id_loss.item())
		epoch_id2ct_loss.update(batch_id2ct_loss.item())
		epoch_ct2id_acc.update(batch_ct2id_acc*100)
		epoch_timer.update(time.time())
		print('\r', epoch_timer, epoch_ct2id_loss, epoch_id2ct_loss, epoch_ct2id_acc, epoch_id2ct_acc, end='       ')
	print('')
	log_dict = {'CT2ID loss': epoch_ct2id_loss.val,
	            'CT2ID ACC': epoch_ct2id_acc.val,
	            'ID2CT loss': epoch_id2ct_loss.val,
	            'ID2CT ACC': epoch_id2ct_acc.val}
	return log_dict


def main():
	# ===========================参数设定===============================
	get_console_args()
	start_epoch = 0
	batch_size = TP['batch_size']
	time_size = math.ceil(40/TP['temporal_gap'])
	merge_win = TP['merge_win']
	merge_size = (time_size-1)//merge_win+1
	alphaI = TP['alphaI']
	alphaC = TP['alphaC']
	dumb_id_label = torch.arange(batch_size)
	dumb_ct_label = torch.arange(merge_size)
	if TP['gpu']:
		dumb_id_label = dumb_id_label.cuda()
		dumb_ct_label = dumb_ct_label.cuda()
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

	# ============================WandB日志=============================
	wandb.init(project='CrossModalityMatch', config=TP)
	wandb.watch(sync_net)

	# ============================度量载入===============================
	epoch_loss_final = Meter('Final Loss', 'avg', ':.2f')
	epoch_id_loss = Meter('ID Loss', 'avg', ':.2f')
	epoch_id_acc = Meter('ID ACC', 'avg', ':.2f', '%,')
	epoch_id_rand_acc = Meter('ID Rand ACC', 'avg', ':.2f', '%,')
	epoch_ct_loss = Meter('CT Loss', 'avg', ':.2f')
	epoch_ct_acc = Meter('CT ACC', 'avg', ':.2f', '%,')
	epoch_ct_rand_acc = Meter('CT Rand ACC', 'avg', ':.2f', '%,')
	epoch_timer = Meter('Time', 'time', ':3.0f')

	epoch_ct_rand_acc.update(100/merge_size)
	epoch_reset_list = [epoch_loss_final, epoch_timer,
	                    epoch_id_loss, epoch_id_acc, epoch_id_rand_acc,
	                    epoch_ct_loss, epoch_ct_acc, ]
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
	valid2_loader = valid_loader.clone(2)
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
			audio_ct, audio_id = sync_net.forward_aud(data_audio)
			video_ct, video_id = sync_net.forward_vid(data_video)
			# (batch_size, feature, time_size)

			# ======================计算身份损失===========================
			audio_random_id = SampleFromTime(audio_id, TP['reduce_method'])
			video_random_id = SampleFromTime(video_id, TP['reduce_method'])
			# (batch_size, feature)

			batch_id_loss, id_score = criterion(dumb_id_label, audio_random_id, video_random_id)
			batch_id_acc = topk_acc(id_score, data_label, 1)
			id_rand_acc = rand_acc(data_label)

			# ======================计算内容损失===========================
			batch_ct_loss = 0
			batch_ct_acc = 0
			for a_merge, v_merge in ContentTransform(audio_ct, video_ct, merge_win, TP['gpu']):
				tmp_ct_loss, tmp_ct_score = criterion(dumb_ct_label, a_merge, v_merge)
				batch_ct_loss += tmp_ct_loss
				# epoch_ct_acc.update(topk_acc(tmp_ct_score, dumb_ct_label)*100)
				batch_ct_acc += topk_acc(tmp_ct_score, dumb_ct_label)/batch_size
			batch_ct_loss /= batch_size

			optimizer.zero_grad()
			batch_loss_final = alphaI*batch_id_loss+alphaC*batch_ct_loss
			batch_loss_final.backward()
			# nn.utils.clip_grad_norm_(sync_net.parameters(), max_norm=10, norm_type=2)
			if TP['affine']:
				print('W grad before clip:', str(affine_net.w.grad))
				print('B grad before clip:', str(affine_net.b.grad))
				nn.utils.clip_grad_value_(affine_net.parameters(), clip_value=1.1)
				# nn.utils.clip_grad_norm_(affine_net.parameters(), max_norm=10, norm_type=2)
				print('W grad after clip:', str(affine_net.w.grad))
				print('B grad after clip:', str(affine_net.b.grad))
			optimizer.step()

			# ==========计量更新============================
			epoch_id_acc.update(batch_id_acc*100)
			epoch_id_rand_acc.update(id_rand_acc*100)
			epoch_id_loss.update(batch_id_loss.item())
			epoch_ct_acc.update(batch_ct_acc*100)
			epoch_ct_loss.update(batch_ct_loss.item())
			epoch_loss_final.update(batch_loss_final.item())
			epoch_timer.update(time.time())
			batch_cnt += 1
			print('\rBatch:(%02d/%d)'%(batch_cnt, len(train_loader)),
			      epoch_timer, epoch_loss_final, epoch_id_loss, epoch_ct_loss,
			      epoch_id_acc, epoch_id_rand_acc,
			      epoch_ct_acc, epoch_ct_rand_acc, end='        ')
		print('')
		print('Epoch:', epoch, epoch_loss_final, epoch_id_loss, epoch_ct_loss,
		      epoch_id_acc, epoch_id_rand_acc,
		      epoch_ct_acc, epoch_ct_rand_acc,
		      file=file_train_log)
		log_dict = {'final loss': epoch_loss_final.avg,
		            'id loss': epoch_id_loss.avg,
		            'ct loss': epoch_ct_loss.avg,
		            'id acc': epoch_id_acc.avg,
		            'ct acc': epoch_ct_acc.avg}
		for meter in epoch_reset_list:
			meter.reset()

		# =======================保存模型=======================
		if TP['gpu']:
			torch.cuda.empty_cache()
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

		# ===========================验证=======================
		valid_log = dict()
		if (epoch+1)%TP['valid_step'] == 0:
			sync_net.eval()
			criterion.eval()
			valid_log = acc_valid(valid_loader, sync_net, criterion, file_train_log)
			valid2_log = acc_valid(valid2_loader, sync_net, criterion, file_train_log)
			sync_net.train()
			criterion.train()
		log_dict.update(valid_log)
		log_dict.update(valid2_log)

		# ========================解耦提取======================
		if (epoch+1)%TP['dis_step'] == 0:
			dis_log = dis_info(train_loader, sync_net, optimizer)
			log_dict.update(dis_log)

		wandb.log(log_dict)
	file_train_log.close()
	wandb.finish()


if __name__ == '__main__':
	main()
