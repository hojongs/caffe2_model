# hojong's caffe2 model
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import brew, core, scope, workspace
from caffe2.python.modeling.parameter_info import ParameterTags
from caffe2.python.model_helper import ModelHelper
from caffe2.python.cnn import CNNModelHelper

import unittest
import numpy as np

m, k, n = (1, 28*28, 10) 							# [m][k] * [k][n] = [m][n]
x = np.random.rand(m, k).astype(np.float32) - 0.5 	# x = m*k 2D tensor

workspace.ResetWorkspace() 							# clear workspace
workspace.FeedBlob("x", x) 							# feed x as a blob
model = ModelHelper(name="test_model") 				# create model

model.Proto() 					# print model's protocol buffer before add operator
brew.fc(model, "x", "y", k, n) 	# fully connected NN, weight = k*n 2D tensor /// bias, y = m*n 2D tensor
brew.softmax(model, "y", "z")
model.Validate()
model.Proto() 					# print model's protocol buffer after add operator


workspace.RunNetOnce(model.param_init_net) 	# init [y_w(weight), y_b(bias) (randomize)]
											# weight is 2D array, bias is 1D array
workspace.Blobs() 							# print workspace's blobs
# workspace.FetchBlob("y_w")
# workspace.FetchBlob("y_b")

workspace.RunNetOnce(model.net)
# y = workspace.FetchBlob("y")
# z = workspace.FetchBlob("z")
