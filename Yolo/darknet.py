import os
import time
import cv2
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *


def get_test_input(im_file):
	img = cv2.imread(im_file)
	img = cv2.resize(img, (416,416))
	img_ = img[:,:,::-1].transpose((2,0,1))
	img_ = img_[np.newaxis,:,:,:]/255.0
	img_ = T.from_numpy(img_).float()
	img_ = Variable(img_)
	return img_


def parse_cfg(cfgfile):
	file = open(cfgfile, 'r')
	lines = file.read().split('\n')
	lines = [x for x in lines if len(x) > 0]
	lines = [x for x in lines if x[0] != '#']
	lines = [x.rstrip().lstrip() for x in lines]

	block = {}
	blocks = []

	for line in lines:
		if line[0] == "[":
			if len(block) != 0:
				blocks.append(block)
				block = {}
			block["type"] = line[1:-1].rstrip()
		else:
			key,value = line.split("=")
			block[key.rstrip()] = value.lstrip()
	blocks.append(block)

	return blocks


def create_modules(blocks):
	net_info = blocks[0]
	module_list = nn.ModuleList()
	prev_filters = 3
	out_filters = []

	for i, b in enumerate(blocks[1:]):
		module = nn.Sequential()
		type = b["type"]

		if type == "convolutional":
			try:
				batch_norm = int(b["batch_normalize"])
				bias = False
			except:
				batch_norm = 0
				bias = True

			filters = int(b["filters"])
			stride = int(b["stride"])
			kernel_size = int(b["size"])
			padding = int(b["pad"])
			activation = b["activation"]

			if padding:
				pad = (kernel_size - 1) // 2
			else:
				pad = 0

			conv = nn.Conv2d(prev_filters, filters, kernel_size, stride,
				pad, bias = bias)
			module.add_module("conv_{0}".format(i), conv)

			if batch_norm:
				bn = nn.BatchNorm2d(filters)
				module.add_module("batch_norm_{0}".format(i), bn)

			if activation == "leaky":
				activn = nn.LeakyReLU(0.1, inplace=True)
				module.add_module("leaky_{0}".format(i), activn)

		elif type == "upsample":
			stride = b["stride"]
			up = nn.Upsample(scale_factor=stride, mode="bilinear",
								align_corners=True)
			module.add_module("upsample_{}".format(i), up)

		elif type == "shortcut":
			shortcut = EmptyLayer()
			module.add_module("shortcut_{}".format(i), shortcut)

		elif type == "route":
			b["layers"] = b["layers"].split(',')

			start = int(b["layers"][0])
			try:
				end = int(b["layers"][1])
			except:
				end = 0

			if start > 0:
				start = start - i
			if end > 0:
				end = end - i

			route = EmptyLayer()
			module.add_module("route_{0}".format(i), route)

			if end < 0:
				filters = out_filters[i + start] + out_filters[i + end]
			else:
				filters= out_filters[i + start]

		elif type == "yolo":
			mask = b["mask"].split(",")
			mask = [int(b) for b in mask]

			anchors = b["anchors"].split(",")
			anchors = [int(a) for a in anchors]
			anchors = [(anchors[j], anchors[j+1]) for j in range(0, len(anchors),2)]
			anchors = [anchors[j] for j in mask]

			detection = DetectionLayer(anchors)
			module.add_module("Detection_{}".format(i), detection)

		out_filters.append(filters)
		prev_filters = filters
		module_list.append(module)

	return net_info, module_list

class DetectionLayer(nn.Module):
	def __init__(self, anchors):
		super(DetectionLayer, self).__init__()
		self.anchors = anchors


class EmptyLayer(nn.Module):
	def __init__(self):
		super(EmptyLayer, self).__init__()


class Darknet(nn.Module):
	def __init__(self, config, weightfile=None):
		super(Darknet, self).__init__()
		self.blocks = parse_cfg(config)
		self.net_info, self.module_list = create_modules(self.blocks)
		if weightfile != None:
			self.load_weights(weightfile)

	def forward(self, x, CUDA):
		outputs = {}
		write = 0

		for i, b in enumerate(self.blocks[1:]):
			type = b["type"]

			if type == "convolutional" or type == "upsample":
				x = self.module_list[i](x)

			elif type == "route":
				layers = b["layers"]
				layers = [int(a) for a in layers]

				if (layers[0]) > 0:
					layers[0] = layers[0] - i

				if len(layers) == 1:
					x = outputs[i + (layers[0])]

				else:
					if (layers[1]) > 0:
						layers[1] = layers[1] - i

					map1 = outputs[i + layers[0]]
					map2 = outputs[i + layers[1]]

					x = torch.cat((map1, map2), 1)

			elif  type == "shortcut":
				from_ = int(b["from"])
				x = outputs[i-1] + outputs[i+from_]

			elif type == 'yolo':
				anchors = self.module_list[i][0].anchors
				inp_dim = int (self.net_info["height"])
				class_num = int (b["classes"])

				x = x.data
				x = predict_transform(x, inp_dim, anchors, class_num, CUDA)
				if not write:
					detections = x
					write = 1
				else:
					detections = torch.cat((detections, x), 1)

			outputs[i] = x
		return detections

	def load_weights(self, weightfile):
	    #Open the weights file
	    fp = open(weightfile, "rb")

	    #The first 5 values are header information
	    # 1. Major version number
	    # 2. Minor Version Number
	    # 3. Subversion number
	    # 4,5. Images seen by the network (during training)
	    header = np.fromfile(fp, dtype = np.int32, count = 5)
	    self.header = torch.from_numpy(header)
	    self.seen = self.header[3]

	    weights = np.fromfile(fp, dtype = np.float32)

	    ptr = 0
	    for i in range(len(self.module_list)):
	        module_type = self.blocks[i + 1]["type"]

	        #If module_type is convolutional load weights
	        #Otherwise ignore.

	        if module_type == "convolutional":
	            model = self.module_list[i]
	            try:
	                batch_normalize = int(self.blocks[i+1]["batch_normalize"])
	            except:
	                batch_normalize = 0

	            conv = model[0]


	            if (batch_normalize):
	                bn = model[1]

	                #Get the number of weights of Batch Norm Layer
	                num_bn_biases = bn.bias.numel()

	                #Load the weights
	                bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
	                ptr += num_bn_biases

	                bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
	                ptr  += num_bn_biases

	                bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
	                ptr  += num_bn_biases

	                bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
	                ptr  += num_bn_biases

	                #Cast the loaded weights into dims of model weights.
	                bn_biases = bn_biases.view_as(bn.bias.data)
	                bn_weights = bn_weights.view_as(bn.weight.data)
	                bn_running_mean = bn_running_mean.view_as(bn.running_mean)
	                bn_running_var = bn_running_var.view_as(bn.running_var)

	                #Copy the data to model
	                bn.bias.data.copy_(bn_biases)
	                bn.weight.data.copy_(bn_weights)
	                bn.running_mean.copy_(bn_running_mean)
	                bn.running_var.copy_(bn_running_var)

	            else:
	                #Number of biases
	                num_biases = conv.bias.numel()

	                #Load the weights
	                conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
	                ptr = ptr + num_biases

	                #reshape the loaded weights according to the dims of the model weights
	                conv_biases = conv_biases.view_as(conv.bias.data)

	                #Finally copy the data
	                conv.bias.data.copy_(conv_biases)

	            #Let us load the weights for the Convolutional layers
	            num_weights = conv.weight.numel()

	            #Do the same as above for weights
	            conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
	            ptr = ptr + num_weights

	            conv_weights = conv_weights.view_as(conv.weight.data)
	            conv.weight.data.copy_(conv_weights)

#im_name = "cars.jpeg"
im_name = "dog-cycle-car.png"

model = Darknet("config.cfg", "weights/yolov3.weights")
inp = get_test_input(im_name)
#start_time = time.time()

pred = model(inp, torch.cuda.is_available())
output = adjust_results(pred,0.5, 85)

img = cv2.imread(im_name)
img = cv2.resize(img, (416,416))
write_result(output, img, (416, 416))

cv2.imwrite("result.png", img)
