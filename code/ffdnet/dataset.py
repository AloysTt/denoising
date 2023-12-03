"""
Dataset related functions

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import os.path
import random
import glob
import string

import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata
from utils import data_augmentation, normalize

def img_to_patches(img, win, stride=1):
	r"""Converts an image to an array of patches.

	Args:
		img: a numpy array containing a CxHxW RGB (C=3) or grayscale (C=1)
			image
		win: size of the output patches
		stride: int. stride
	"""
	k = 0
	endc = img.shape[0]
	endw = img.shape[1]
	endh = img.shape[2]
	patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
	total_pat_num = patch.shape[1] * patch.shape[2]
	res = np.zeros([endc, win*win, total_pat_num], np.float32)
	for i in range(win):
		for j in range(win):
			patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
			res[:, k, :] = np.array(patch[:]).reshape(endc, total_pat_num)
			k = k + 1
	return res.reshape([endc, win, win, total_pat_num])

def prepare_data(data_path, \
				 val_data_path, \
				 gd_path, \
				 patch_size, \
				 stride, \
				 max_num_patches=None, \
				 aug_times=1, \
				 gray_mode=False):
	r"""Builds the training and validations datasets by scanning the
	corresponding directories for images and extracting	patches from them.

	Args:
		data_path: path containing the training image dataset
		val_data_path: path containing the validation image dataset
		patch_size: size of the patches to extract from the images
		stride: size of stride to extract patches
		stride: size of stride to extract patches
		max_num_patches: maximum number of patches to extract
		aug_times: number of times to augment the available data minus one
		gray_mode: build the databases composed of grayscale patches
	"""
	# training database
	print('> Training database')
	scales = [1, 0.9, 0.8, 0.7]
	types = ('*.bmp', '*.png')
	files = []
	filesgd = []
	for tp in types:
		files.extend(glob.glob(os.path.join(data_path, tp)))
	files.sort()

	for tp in types:
		filesgd.extend(glob.glob(os.path.join(gd_path, tp)))
	filesgd.sort()

	gdMap = {}
	for file in filesgd:
		basename = file.split(os.path.sep)[-1][:-4]
		gdMap[basename] = file

	if gray_mode:
		traindbf = 'train_gray.h5'
		valdbf = 'val_gray.h5'
	else:
		traindbf = 'train_rgb.h5'
		valdbf = 'val_rgb.h5'

	if max_num_patches is None:
		max_num_patches = 5000000
		print("\tMaximum number of patches not set")
	else:
		print("\tMaximum number of patches set to {}".format(max_num_patches))
	train_num = 0
	i = 0
	with h5py.File(traindbf, 'w') as h5f:
		ds = h5f.create_dataset("train", (0, 6, patch_size, patch_size), maxshape=(None, 6, patch_size, patch_size))
		while i < len(files) and train_num < max_num_patches:
			filename = files[i]
			basename = filename.split(os.path.sep)[-1][:-4].rstrip(string.digits).rstrip("_")
			gdName = gdMap[basename]
			imgor = cv2.imread(files[i])
			imgor_gd = cv2.imread(gdName)
			# h, w, c = img.shape
			for sca in scales:
				img = cv2.resize(imgor, (0, 0), fx=sca, fy=sca, \
								interpolation=cv2.INTER_CUBIC)
				img_gd = cv2.resize(imgor_gd, (0, 0), fx=sca, fy=sca, \
								interpolation=cv2.INTER_CUBIC)
				if not gray_mode:
					# CxHxW RGB image
					img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
					img_gd = (cv2.cvtColor(img_gd, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
				else:
					# CxHxW grayscale image (C=1)
					img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					img = np.expand_dims(img, 0)
					img_gd = cv2.cvtColor(img_gd, cv2.COLOR_BGR2GRAY)
					img_gd = np.expand_dims(img_gd, 0)
				img = normalize(img)
				img_gd = normalize(img_gd)
				patches = img_to_patches(img, win=patch_size, stride=stride)
				patches_gd = img_to_patches(img_gd, win=patch_size, stride=stride)
				print("\tfile: %s scale %.1f # samples: %d" % \
					  (files[i], sca, patches.shape[3]*aug_times))
				ds.resize(size=ds.shape[0]+(patches.shape[3]*aug_times), axis=0)
				for nx in range(patches.shape[3]):
					randseed = np.random.randint(0, 7)
					data = data_augmentation(patches[:, :, :, nx].copy(), randseed)
					data_gd = data_augmentation(patches_gd[:, :, :, nx].copy(), randseed)
					data_final = np.concatenate([data, data_gd], axis=0)
					ds[train_num, :, :, :] = data_final
					# h5f.create_dataset(str(train_num), data=data_final)
					train_num += 1
					for mx in range(aug_times-1):
						data_aug = data_augmentation(data_final, np.random.randint(1, 4))
						ds[train_num, :, :, :] = data_aug
						# h5f.create_dataset(str(train_num)+"_aug_%d" % (mx+1), data=data_aug)
						train_num += 1
			i += 1

	# validation database
	print('\n> Validation database')
	files = []
	for tp in types:
		files.extend(glob.glob(os.path.join(val_data_path, tp)))
	files.sort()
	h5f = h5py.File(valdbf, 'w')
	val_num = 0
	for i, item in enumerate(files):
		print("\tfile: %s" % item)
		basename = item.split(os.path.sep)[-1][:-4].rstrip(string.digits).rstrip("_")
		gdName = gdMap[basename]
		img = cv2.imread(item)
		img_gd = cv2.imread(gdName)
		if not gray_mode:
			# C. H. W, RGB image
			img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
			img_gd = (cv2.cvtColor(img_gd, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
		else:
			# C, H, W grayscale image (C=1)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = np.expand_dims(img, 0)
			img_gd = cv2.cvtColor(img_gd, cv2.COLOR_BGR2GRAY)
			img_gd = np.expand_dims(img_gd, 0)
		img = normalize(img)
		img_gd = normalize(img_gd)
		img_final = np.concatenate([img, img_gd], axis=0)
		h5f.create_dataset(str(val_num), data=img_final)
		val_num += 1
	h5f.close()

	print('\n> Total')
	print('\ttraining set, # samples %d' % train_num)
	print('\tvalidation set, # samples %d\n' % val_num)


class DatasetTrain(udata.Dataset):
	r"""Implements torch.utils.data.Dataset
	"""
	def __init__(self, gray_mode=False, shuffle=False):
		super(DatasetTrain, self).__init__()
		self.gray_mode = gray_mode
		if not self.gray_mode:
			self.dbf = 'train_rgb.h5'
		else:
			self.dbf = 'train_gray.h5'

		h5f = h5py.File(self.dbf, 'r')
		self.data = np.zeros(shape=h5f["train"].shape)
		h5f["train"].read_direct(self.data)
		self.keys = list(range(self.data.shape[0]))
		if shuffle:
			random.shuffle(self.keys)

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):
		key = self.keys[index]
		return torch.Tensor(self.data[key, :, :, :])

class DatasetValidation(udata.Dataset):
	r"""Implements torch.utils.data.Dataset
	"""
	def __init__(self, gray_mode=False, shuffle=False):
		super(DatasetValidation, self).__init__()
		self.gray_mode = gray_mode
		if not self.gray_mode:
			self.dbf = 'val_rgb.h5'
		else:
			self.dbf = 'val_gray.h5'

		h5f = h5py.File(self.dbf, 'r')
		self.keys = list(h5f.keys())
		if shuffle:
			random.shuffle(self.keys)
		h5f.close()

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):
		h5f = h5py.File(self.dbf, 'r')
		key = self.keys[index]
		data = np.array(h5f[key])
		h5f.close()
		return torch.Tensor(data)
