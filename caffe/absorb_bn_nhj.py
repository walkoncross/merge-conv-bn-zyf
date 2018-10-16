import sys
sys.path.insert(0,'/root/data/caffe/python')
import caffe
import numpy as np

from google.protobuf import text_format
from caffe.proto import caffe_pb2

dirname = './'

origin_deploy_filename = dirname + 'quality_v0.0.1.prototxt'
origin_net_model_filename = dirname + 'quality_v0.0.1.caffemodel'

deploy_filename = dirname + 'quality_v0.0.1_nobn.prototxt'
net_model_filename = dirname + 'quality_v0.0.1_nobn.caffemodel'

caffe.set_mode_cpu()

origin_net_model = caffe.Net(origin_deploy_filename,origin_net_model_filename,caffe.TEST)
net_model = caffe.Net(deploy_filename,caffe.TEST)

#
origin_net = caffe_pb2.NetParameter()
try:
    f = open(origin_deploy_filename,'rb')
    text_format.Parse(f.read(),origin_net)
    f.close()
except IOError:
    exit('Could not open file ' + origin_deploy_filename)
#

for index in xrange(0,len(origin_net.layer)):
	layer = origin_net.layer[index]
	if layer.type == 'Convolution':
		origin_weight = origin_net_model.params[layer.name][0].data
		
		weight = net_model.params[layer.name][0].data
		bias = net_model.params[layer.name][1].data

		if origin_net.layer[index+1].type == 'BatchNorm':
			# BatchNorm
			eps = 1e-5
			bn_layer_name = origin_net.layer[index+1].name
			mean = origin_net_model.params[bn_layer_name][0].data / origin_net_model.params[bn_layer_name][2].data
			var = origin_net_model.params[bn_layer_name][1].data / origin_net_model.params[bn_layer_name][2].data
			var = np.sqrt(var + eps)
			# Scale
			scale_layer_name = origin_net.layer[index+2].name
			scale = origin_net_model.params[scale_layer_name][0].data
			scale_bias = origin_net_model.params[scale_layer_name][1].data
			#
			#
			num_kernel = origin_weight.shape[0]
			
			for kernel_index in xrange(0,num_kernel):
				weight[kernel_index,:,:,:] = scale[kernel_index] * origin_weight[kernel_index,:,:,:] / var[kernel_index]
				bias[kernel_index] = scale_bias[kernel_index] - scale[kernel_index] * mean[kernel_index] / var[kernel_index]
		if len(origin_net_model.params[layer.name]) == 2:
			#
			print(layer.name,bn_layer_name,scale_layer_name)
			#
			origin_bias = net_model.params[layer.name][1].data
			for kernel_index in xrange(0,num_kernel):
				bias[kernel_index] = bias[kernel_index] + scale[kernel_index]*origin_bias[kernel_index]/var[kernel_index]
	elif layer.type == 'InnerProduct':
		net_model.params[layer.name][0].data[:] = origin_net_model.params[layer.name][0].data[:]
		net_model.params[layer.name][1].data[:] = origin_net_model.params[layer.name][1].data[:]
	elif layer.type == 'PReLU':
		net_model.params[layer.name][0].data[:] = origin_net_model.params[layer.name][0].data[:]

net_model.save(net_model_filename)

net_model.blobs['data'].data[:] = 0
net_model.forward()

origin_net_model.blobs['data'].data[:] = 0
origin_net_model.forward()

print(np.sum(np.abs(origin_net_model.blobs['conv1/bn'].data - net_model.blobs['conv1'].data)))
print(np.sum(np.abs(origin_net_model.blobs['prob'].data - net_model.blobs['prob'].data)))




