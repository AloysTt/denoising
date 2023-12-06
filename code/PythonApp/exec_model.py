import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import ImageTk, Image
from torch.autograd import Variable
from models import FFDNet
from utils import normalize, remove_dataparallel_wrapper, is_rgb, variable_to_cv2_image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_model(image: cv2.typing.MatLike, cuda: bool, model_path: str, sigma: float):
	try:
		rgb_den = is_rgb(image)
	except:
		raise Exception('Could not open the input image')

	sigma /= 255.

	# Open image as a CxHxW torch.Tensor
	in_ch = 3
	# from HxWxC to CxHxW, RGB image
	if image.shape[2] == 4:
		imorig = image[:, :, :3].transpose(2, 0, 1)
	else:
		imorig = image.transpose(2, 0, 1)
	imorig = np.expand_dims(imorig, 0)

	# Handle odd sizes
	expanded_h = False
	expanded_w = False
	sh_im = imorig.shape
	if sh_im[2]%2 == 1:
		expanded_h = True
		imorig = np.concatenate((imorig, \
				imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

	if sh_im[3]%2 == 1:
		expanded_w = True
		imorig = np.concatenate((imorig, \
				imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

	imorig = normalize(imorig)
	imorig = torch.Tensor(imorig)

	# Create model
	print('Loading model ...\n')
	net = FFDNet(num_input_channels=in_ch)

	# Load saved weights
	if cuda:
		state_dict = torch.load(model_path)
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).cuda()
	else:
		state_dict = torch.load(model_path, map_location='cpu')
		# CPU mode: remove the DataParallel wrapper
		state_dict = remove_dataparallel_wrapper(state_dict)
		model = net
	model.load_state_dict(state_dict)

	# Sets the model in evaluation mode (e.g. it removes BN)
	model.eval()

	# Sets data type according to CPU or GPU modes
	if cuda:
		dtype = torch.cuda.FloatTensor
	else:
		dtype = torch.FloatTensor

	# Test mode
	with torch.no_grad(): # PyTorch v0.4.0
		imnoisy = Variable(imorig.type(dtype))
		nsigma = Variable(
				torch.FloatTensor([sigma]).type(dtype))

	# Estimate noise and subtract it to the input image
	im_denoised = model(imnoisy, nsigma)
	outim = im_denoised

	if expanded_h:
		imorig = imorig[:, :, :-1, :]
		outim = outim[:, :, :-1, :]
		imnoisy = imnoisy[:, :, :-1, :]

	if expanded_w:
		imorig = imorig[:, :, :, :-1]
		outim = outim[:, :, :, :-1]
		imnoisy = imnoisy[:, :, :, :-1]

	return np.array(variable_to_cv2_image(outim))
