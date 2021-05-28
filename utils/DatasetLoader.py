import torch
import numpy as np
import random
import pdb
import os
import threading
import time
from queue import Queue
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.GetDataFromFile import loadWAV, get_frames


class MyDataset(Dataset):
	def __init__(self, dataset_file_name, unique_mode=False, maxFrames=44):
		self.dataset_file_name = dataset_file_name
		self.info_list = []
		self.data_list = []
		self.ndata = None
		self.eval_mode = unique_mode
		self.maxFrames = maxFrames

		check_set = set()
		with open(dataset_file_name) as listfile:
			while True:
				line = listfile.readline()
				if not line:
					break
				data = line.split()
				# if len(data) == 4:
				if len(data) == 5:
					if abs(int(data[3]))-abs(int(data[2]))>=maxFrames+4:
						if unique_mode:
							if int(data[-1]) not in check_set:
								check_set.add(int(data[-1]))
							self.info_list.append(data)
						else:
							self.info_list.append(data)
					else:
						print('%s is too short'%(data[0]))
				else:
					raise
		for info in tqdm(self.info_list):
			mp4name, wavname = info[0], info[1]
			mp4data = get_frames(mp4name, max_frames=self.maxFrames, start_frame=0)
			wavdata = loadWAV(wavname, max_frames=self.maxFrames*4, start_frame=0)
			mp4data, wavdata = mp4data.squeeze(), wavdata.squeeze()
			self.data_list.append((mp4data, wavdata, info[-1]))
		self.ndata = len(self.data_list)
		print('Unique Mode %s - %d clips'%(self.eval_mode, len(self.data_list)))

	def __getitem__(self, item):
		return self.data_list[item]

	def __len__(self):
		return self.ndata


class MyDataLoader(DataLoader):
	def __init__(self, dataset_file_name, batch_size, num_workers, eval_mode=False, **kwargs):
		self.dataset = MyDataset(dataset_file_name, eval_mode)
		self.num_workers = num_workers
		self.batch_size = batch_size
		super().__init__(self.dataset, shuffle=True, batch_size=batch_size,
		                 drop_last=True, num_workers=num_workers)
		self.nFiles = len(self.dataset)

	def clone(self, batch_size, **kwargs):
		clone_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True,
		                          drop_last=True, num_workers=self.num_workers)
		return clone_loader


class MultiLoader(object):
	def __init__(self, dataset_file_name, nPerEpoch, nBatchSize, maxFrames,
	             nDataLoaderThread, maxQueueSize=10, evalmode=False, **kwargs):
		self.dataset_file_name = dataset_file_name
		self.nPerEpoch = nPerEpoch
		self.nWorkers = nDataLoaderThread
		self.nMaxFrames = maxFrames
		self.batch_size = nBatchSize
		self.maxQueueSize = maxQueueSize

		self.data_list = []
		self.data_epoch = []

		self.nFiles = 0
		self.evalmode = evalmode

		self.dataLoaders = []

		with open(dataset_file_name) as listfile:
			while True:
				line = listfile.readline()
				if not line:
					break

				data = line.split()

				if len(data) == 5:
					if abs(int(data[3]))-abs(int(data[2]))>=maxFrames+4:
						self.data_list.append(data)
					else:
						print('%s is too short'%(data[0]))
				else:
					raise

		# Initialize Workers...
		self.datasetQueue = Queue(self.maxQueueSize)

		print('Evalmode %s - %d clips'%(self.evalmode, len(self.data_list)))

	def dataLoaderThread(self, nThreadIndex):

		index = nThreadIndex*self.batch_size

		if index>=self.nFiles:
			return

		while True:
			if self.datasetQueue.full():
				time.sleep(1.0)
				continue

			feat_a = []
			feat_v = []
			feat_id = []

			for filename in self.data_epoch[index:index+self.batch_size]:
				people_cnt = int(filename[-1])
				feat_a.append(loadWAV(filename[1], max_frames=self.nMaxFrames*4))
				feat_v.append(get_frames(filename[0], max_frames=self.nMaxFrames))
				feat_id.append(people_cnt)

			data_video = torch.cat(feat_v, dim=0)
			data_audio = torch.cat(feat_a, dim=0)
			data_id = np.stack(feat_id)

			self.datasetQueue.put([data_video, data_audio, data_id])

			index += self.batch_size*self.nWorkers

			if index+self.batch_size>self.nFiles:
				break

	def __iter__(self):
		# Shuffle one more
		random.shuffle(self.data_list)

		self.data_epoch = self.data_list[:min(self.nPerEpoch, len(self.data_list))]
		self.nFiles = len(self.data_epoch)

		# Make and Execute Threads...
		for index in range(0, self.nWorkers):
			self.dataLoaders.append(threading.Thread(target=self.dataLoaderThread, args=[index]))
			self.dataLoaders[-1].start()

		return self

	def __next__(self):
		while True:
			isFinished = True

			if not self.datasetQueue.empty():
				return self.datasetQueue.get()
			for index in range(0, self.nWorkers):
				if self.dataLoaders[index].is_alive():
					isFinished = False
					break

			if not isFinished:
				time.sleep(1.0)
				continue

			for index in range(0, self.nWorkers):
				self.dataLoaders[index].join()

			self.dataLoaders = []
			raise StopIteration

	def __call__(self):
		pass

	def getDatasetName(self):
		return self.dataset_file_name

	def qsize(self):
		return self.datasetQueue.qsize()
