import os
import cv2
import glob
import tqdm


def count_mp4(filename):
	cap = cv2.VideoCapture(filename)
	total_frames = cap.get(7)
	return [total_frames, total_frames]


def get_meta(train_dir='train.txt', test_dir='test.txt'):
	peoplelist = os.listdir('/hdd_dataset/2/2_vox2/1_mp4')
	print('total people: ', len(peoplelist))
	train_cnt = len(peoplelist)*0.9
	with open(train_dir, 'w') as ftrain, open(test_dir, 'w') as ftest:
		for i, people in tqdm.tqdm(enumerate(peoplelist)):
			mp4list = glob.glob('/hdd_dataset/2/2_vox2/1_mp4/'+people+'/*/*.mp4')
			for mp4path in mp4list:
				wavpath = mp4path.replace('1_mp4', '2_wav')
				wavpath = wavpath.replace('.mp4', '.wav')
				# print(mp4path)
				# print(wavpath)
				# exit()
				if os.path.exists(wavpath):
					total_frames = count_mp4(mp4path)
					if total_frames[0] != total_frames[1]:
						print('ERROR, frames mismatch:', total_frames[0], total_frames[1])
						continue
					if i<train_cnt:
						ftrain.write("%s %s %s %d\n"%(mp4path, wavpath, '0', total_frames[0]))
					else:
						ftest.write("%s %s %s %d\n"%(mp4path, wavpath, '0', total_frames[0]))


if __name__ == '__main__':
	get_meta('./data/train.txt', './data/test.txt')
