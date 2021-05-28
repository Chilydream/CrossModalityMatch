# import wandb
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# import wandb

from criterion.CrossEntropy import CrossEntropy
from criterion.InfoNCE import InfoNCE
from model.AffineModel import AffineModel
from utils.DatasetLoader import MyDataLoader
from config.TrainConfig import TRAIN_PARAMETER as TP


maxFrames = 40
people_dict = dict()
people_cnt = 0
people_repeat = dict()


with open('./data/train.txt', 'r') as fr, open('./data/train_part.txt', 'w') as fw:
	lines = fr.readlines()
	for line in lines:
		data = line.split()
		people = data[0].strip().split('/')[-3]
		if abs(int(data[3]))-abs(int(data[2]))>=maxFrames+4:
			if people not in people_dict.keys():
				people_dict[people] = people_cnt
				people_repeat[people] = 1
				print(line, people_cnt, sep=' ', end='', file=fw)
				people_cnt += 1
			else:
				if people_repeat[people]>5:
					continue
				data.append(people_dict[people])
				people_repeat[people] += 1
				print(line, people_dict[people], sep=' ', end='', file=fw)
		else:
			print('%s is too short'%(data[0]))


with open('./data/test.txt', 'r') as fr, open('./data/test_part.txt', 'w') as fw:
	lines = fr.readlines()
	for line in lines:
		data = line.split()
		people = data[0].strip().split('/')[-3]
		if abs(int(data[3]))-abs(int(data[2]))>=maxFrames+4:
			if people not in people_dict.keys():
				people_dict[people] = people_cnt
				people_repeat[people] = 1
				print(line, people_cnt, sep=' ', end='', file=fw)
				people_cnt += 1
			else:
				if people_repeat[people]>5:
					continue
				data.append(people_dict[people])
				people_repeat[people] += 1
				print(line, people_dict[people], sep=' ', end='', file=fw)
		else:
			print('%s is too short'%(data[0]))

# train_loader = MyDataLoader(TP['train_list'], 30, TP['num_workers'])
# valid_loader = MyDataLoader(TP['test_list'], 30, TP['num_workers'])
