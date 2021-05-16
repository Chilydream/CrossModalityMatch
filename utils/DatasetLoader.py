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
	def __init__(self, dataset_file_name, eval_mode=False, maxFrames=40):
		self.dataset_file_name = dataset_file_name
		self.info_list = []
		self.data_list = []
		self.ndata = None
		self.eval_mode = eval_mode
		self.maxFrames = maxFrames

		people_dict = dict()
		people_cnt = 0
		with open(dataset_file_name) as listfile:
			while True:
				line = listfile.readline()
				if not line:
					break
				data = line.split()
				people = data[0].strip().split('/')[-2]
				if len(data) == 4:
					if abs(int(data[3]))-abs(int(data[2]))>=maxFrames+4:
						if people not in people_dict.keys():
							people_dict[people] = people_cnt
							people_cnt += 1
							data.append(people_cnt)
							if eval_mode:
								self.info_list.append(data)
						else:
							data.append(people_dict[people])
						if not eval_mode:
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
		print('Evalmode %s - %d clips'%(self.eval_mode, len(self.data_list)))

	def __getitem__(self, item):
		return self.data_list[item]

	def __len__(self):
		return self.ndata


class MyDataLoader(DataLoader):
	def __init__(self, dataset_file_name, batch_size, num_workers, eval_mode=False, **kwargs):
		self.dataset = MyDataset(dataset_file_name, eval_mode)
		super().__init__(self.dataset, shuffle=True, batch_size=batch_size,
		                 drop_last=True, num_workers=num_workers)
		self.nFiles = len(self.dataset)

