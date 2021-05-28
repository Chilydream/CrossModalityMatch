import torch
import numpy as np
import random
import pdb
import os
import cv2
import math
from scipy.io import wavfile


def loadWAV(filename, max_frames, start_frame=0, evalmode=False, num_eval=10):
	npy_name = filename+'.npy'
	npy_name = npy_name.replace('/', '_').replace('\\', '_')
	npy_name = os.path.join('/hdd_data/fsx/CrossModalityMatch/cache', npy_name)
	if os.path.exists(npy_name):
		feat = np.load(npy_name, allow_pickle=True)
	else:
		# Maximum audio length
		max_audio = max_frames*160+240
		# self.nMaxFrames = 40 的情况下
		# max_audio = 25840
		# self.nMaxFrames = 50 的情况下
		# max_audio = 32240
		start_audio = start_frame*160
		# ques: 为什么传进来的 start frame要乘 4

		# Read wav file and convert to torch tensor
		sample_rate, audio = wavfile.read(filename)
		# audio.shape = (88751)
		# 如果是双声道：audio.shape = (88751, 2)

		audiosize = audio.shape[0]

		if audiosize<=max_audio:
			raise ValueError('Audio clip is too short')

		if evalmode:
			start_frame = np.linspace(0, audiosize-max_audio, num=num_eval)
		else:
			start_frame = np.array([start_audio])

		feats = []
		for asf in start_frame:
			feats.append(audio[int(asf):int(asf)+max_audio])

		feat = np.stack(feats, axis=0)
		np.save(npy_name, feat)

	feat = torch.FloatTensor(feat)
	# feat size = [1, max_audio=25840]

	# 返回的是（1，max_audio）的 tensor
	return feat


def make_image_square(img):
	s = max(img.shape[0:2])
	f = np.zeros((s, s, 3), np.uint8)
	ax, ay = (s-img.shape[1])//2, (s-img.shape[0])//2
	f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img
	return f


def get_frames(filename, max_frames=100, start_frame=0):
	npy_name = filename+'.npy'
	npy_name = npy_name.replace('/', '_').replace('\\', '_')
	npy_name = os.path.join('/hdd_data/fsx/CrossModalityMatch/cache', npy_name)
	if os.path.exists(npy_name):
		im = np.load(npy_name, allow_pickle=True)
	else:
		cap = cv2.VideoCapture(filename)

		cap.set(1, start_frame)

		images = []
		for frame_num in range(0, max_frames):
			ret, image = cap.read()
			image = make_image_square(image)
			image = cv2.resize(image, (240, 240))
			images.append(image)

		cap.release()
		# im的形状是（w，h，3）
		im = np.stack(images, axis=3)
		# stack操作后 im的形状是（w，h，3，max_frames）
		im = np.expand_dims(im, axis=0)
		# im的形状是（1，w，h，3，max_frames)
		im = np.transpose(im, (0, 3, 4, 1, 2))
		# im的形状是（1，3，max_frames，w，h）
		np.save(npy_name, im)
	imtv = torch.FloatTensor(im)

	return imtv
