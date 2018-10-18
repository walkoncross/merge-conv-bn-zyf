#coding=utf-8
 
import os.path as osp
import sys
import copy
import os
from sys import path
import numpy as np
import google.protobuf as pb
 
#path.append('/data1/henryzhong/caffe/python')
path.append('/data2/nfs_share/caffe/python')
print (path)
import argparse
 
import caffe
import caffe.proto.caffe_pb2 as cp
 
caffe.set_mode_cpu()
layer_type = ['Convolution', 'InnerProduct']
bnn_type = ['BatchNorm', 'Scale']
temp_file = './temp.prototxt'
 
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='convert prototxt to prototxt without batch normalization')
    parser.add_argument('--model', dest='caffe_config_filename',
                        help='prototxt file',
                        default="./models/deploy_68_new.prototxt", 
                        type=str)
    parser.add_argument('--weights', dest='caffe_weights_filename',
                        help='weights file',
                        default="./models/vgg_68_new.caffemodel", 
                        type=str)
    parser.add_argument('--merged-model',dest='caffe_without_bn_config_filename',
                        help='mobile config file',
                        default="./models/result.prototxt",
                        type=str)
    parser.add_argument('--merged-weights',dest='caffe_without_bn_weight_filename',
                        help='mobile weights file',
                        default="./models/result.caffemodel",
                        type=str)
 
    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    #
    args = parser.parse_args()
    return args
 
class ConvertBnn:
    def __init__(self, model, weights, dest_model_dir, dest_weight_dir):
        self.net_model = caffe.Net(model, weights, caffe.TEST)
        self.net_param = self.get_netparameter(model)
        self.dest_model = None
        self.dest_param = self.get_netparameter(model)
        self.remove_ele = []
        self.bnn_layer_location = []
        self.dest_dir = dest_model_dir
        self.dest_weight_dir = dest_weight_dir
        self.pre_process()
         
    def pre_process(self):
        net_param = self.dest_param
        layer_params = net_param.layer
        length = len(layer_params)
        i = 0
        while i < length:
            print ('layer: ', i)
             
            if layer_params[i].type in layer_type:
                if (i + 2 < length) and layer_params[i + 1].type == bnn_type[0] and  \
                    layer_params[i + 2].type == bnn_type[1]:
                        params = layer_params[i].param
                        if len(params) ==0:
                            params.add()
                            params[0].lr_mult = 1
                            params[0].decay_mult = 1
                        if len(params) < 2:
                            params.add()
                            params[1].lr_mult = 2
                            params[1].decay_mult = 0
                            if layer_params[i].type in ['Convolution']:
                                layer_params[i].convolution_param.bias_term = True
                                layer_params[i].convolution_param.bias_filler.type = 'constant'
                                layer_params[i].convolution_param.bias_filler.value = 0
                            elif layer_params[i].type == 'InnerProduct':
                                layer_params[i].inner_product_param.bias_term = True
                                layer_params[i].inner_product_param.bias_filler.type = 'constant'
                                layer_params[i].inner_product_param.bias_filler.value = 0
                        #修改配置params
                        self.bnn_layer_location.extend([i])
                        self.remove_ele.extend([layer_params[i + 1], layer_params[i + 2]])
                        i = i + 3
                else:
                    i=i+1
            elif layer_params[i].type == 'Scale' and layer_params[i-1].type == bnn_type[0]:
                self.bnn_layer_location.extend([i])
                self.remove_ele.extend([layer_params[i -1]])
                i += 1
            else:
                i += 1
        #for ele in remove_ele:
        #    layer_params.remove(ele)
        with open(temp_file, 'w') as f:
            f.write(str(net_param))
        print ('Find bn position and remove bn, scale layer')
        self.dest_model = caffe.Net(temp_file, caffe.TEST)
        model_layers = self.net_model.layers
        for i, layer in enumerate(model_layers):
            if layer.type == layer_type[0] or layer.type == layer_type[1] \
                    or layer.type == bnn_type[1]:
                self.dest_model.layers[i].blobs[0] = layer.blobs[0]
                if len(layer.blobs) > 1:
                    self.dest_model.layers[i].blobs[1] = layer.blobs[1]
        print ('Add weights end')
     
    def get_netparameter(self, model):
        with open(model) as f:
            net = cp.NetParameter()
            pb.text_format.Parse(f.read(), net)
            return net
 
    def convert(self):
        out_params = self.dest_param.layer
        model_layers = self.net_model.layers
        out_model_layers = self.dest_model.layers
         
        length = len(self.bnn_layer_location)
        param_layers = self.dest_param.layer
        
        '''
        print ('output: layer name')
        for layer in param_layers:
            print (layer.name)
 
        print ('*******************************')
        
        print ('output: layer type')
        for layer in self.net_model.layers:
            print (layer.type)
 
        param_layer_type_list = [layer.type for layer in param_layers]
        model_layer_type_list = [layer.type for layer in model_layers]
 
        i=j=0
        dict_layer_id_param_to_model = {}
        while i < len(param_layer_type_list):
            if param_layer_type_list[i]==model_layer_type_list[j]:
                dict_layer_id_param_to_model[i]=j
                i=i+1
                j=j+1
            else:
                j=j+1
        print (dict_layer_id_param_to_model)
        '''

        print ('Merge bn weights into conv or fc or depthwise')
        l = 0
        while l < length:
            i = self.bnn_layer_location[l]
            print (param_layers[i].name, param_layers[i].type)
            if param_layers[i].type in layer_type:
                #i = self.net_model.params.keys().index(param_layers[l].name);
                 
                channels = self.net_model.params[param_layers[i].name][0].num
                # bn layer weights
                eps = 2e-5
                moving_average = self.net_model.params[param_layers[i + 1].name][2].data[0]
                if moving_average != 0:
                    scale_factor = 1 / moving_average
                else:
                    scale_factor = 0
                mean = scale_factor * self.net_model.params[param_layers[i+1].name][0].data
                std = np.sqrt(scale_factor * self.net_model.params[param_layers[i+1].name][1].data + eps)
                # scale layer weights
                scale_weight = self.net_model.params[param_layers[i+2].name][0].data
                scale_bias = self.net_model.params[param_layers[i+2].name][1].data
                # introduce alpha to simply computetion
                alpha = scale_weight / std
                for k in range(channels):
                    self.dest_model.params[param_layers[i].name][0].data[k] = self.net_model.params[param_layers[i].name][0].data[k] * alpha[k]
                    self.dest_model.params[param_layers[i].name][1].data[k] = self.dest_model.params[param_layers[i].name][1].data[k] * alpha[k] + (scale_bias[k] - mean[k] *alpha[k])
            # TODO
            elif param_layers[i].type == 'Scale':
                print('Scale')
                channels = self.net_model.params[param_layers[i-1].name][0].num
                moving_average = self.net_model.params[param_layers[i - 1].name][2].data[0]
                if moving_average != 0:
                    scale_factor = 1 / moving_average
                else:
                    scale_factor = 0
                eps = 2e-5
                mean = scale_factor * self.net_model.params[param_layers[i-1].name][0].data
                std = np.sqrt(scale_factor * self.net_model.params[param_layers[i-1].name][1].data + eps)

                a = copy.deepcopy(self.net_model.params[param_layers[i].name][0].data)
                b = copy.deepcopy(self.net_model.params[param_layers[i].name][1].data)
                for k in range(channels):
                    self.dest_model.params[param_layers[i].name][0].data[k] = a[k] / std[k]
                    self.dest_model.params[param_layers[i].name][1].data[k] = a[k] / std[k] - a[k] * mean[k] / std[k] + b[k]
            l += 1
        self.dest_model.save(self.dest_weight_dir)
        for ele in self.remove_ele:
            out_params.remove(ele)
        with open(self.dest_dir, 'w') as f:
            f.write(str(self.dest_param))
        os.remove(temp_file)
        print ('MERGED SUCCEED!')
 
if __name__ == '__main__':
    args = parse_args()
    cb = ConvertBnn(args.caffe_config_filename,args.caffe_weights_filename,args.caffe_without_bn_config_filename,args.caffe_without_bn_weight_filename)
    cb.convert()
