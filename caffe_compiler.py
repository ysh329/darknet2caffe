#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from  caffe_pb2 import * 
import argparse

def compile_caffe(src_dir, dst_dir, src_file):
  os.system('protoc -I={} --python_out={} {}'.
            format(src_dir, dst_dir, src_file))


def read_caffemodel(src_file):
  with open(src_file, 'rb') as f:
    #caffemodel = caffe_pb2.NetParameter()
    caffemodel = NetParameter()
    caffemodel.ParseFromString(f.read())
#    for item in caffemodel.layers#:
#      print(item)
    if(len(caffemodel.layer)==0):
	if(len(caffemodel.layers)==0):
            print " the caffemodel is null!"
        else:
            layermodel = caffemodel.layers
    else:
         layermodel = caffemodel.layer

    import re
    model_name_pattern = "(.*).caffemodel"
    print(src_file)
    model_name = re.findall(model_name_pattern, src_file)[0]
    save_caffemodel_data(layermodel, model_name)

def get_shape_data(blobs):
    blobs_num = len(blobs)
    shape_data =[]
    shape_data.insert(0,blobs_num)
    for blob in blobs:
        shape = blob.shape
        length = len(shape.dim)
        tmpdata=[]
        if(length==0):
            tmpdata.append(blob.num)
	    tmpdata.append(blob.channels)
	    tmpdata.append(blob.height)
	    tmpdata.append(blob.width)
            for tmp in tmpdata:
                if(tmp>=1):
                    length+=1
  
        shape_data.append(length)
    for blob in blobs:
        shape = blob.shape
        if(len(shape.dim)>0):
            shape_data.extend(shape.dim)
	else:
            shape_data.append(blob.num)
            shape_data.append(blob.channels)
            shape_data.append(blob.height)
            shape_data.append(blob.width)
    return shape_data 

def save_caffemodel_data(model_layer, model_name):

    fo = open("{}.dat".format(model_name), "w")
    """ get number of layer has blob """
    item_num=0
    for item in model_layer:
        blobs = item.blobs
        if(len(blobs))>0:
            item_num+=1
    """ write  the number of layer has blob into file """
    fo.write( (100 - len(str(item_num)))*' ' + str(item_num)+'\n')
    
    for item_idx in xrange(len(model_layer)):
        item = model_layer[item_idx]
        blobs=item.blobs
        if(len(blobs)>0):
            fo.write(item.name + ' ')
	    shape_data = get_shape_data(blobs)
            print("{}\t{}".format(item_idx, str(shape_data)))
            for meta_data in shape_data:
                fo.write(str(meta_data) +' ')
	    for blob in blobs:
                data = blob.data
                for weight_data in data:
                    fo.write(str(round(weight_data,6)) + ' ')
            fo.write('\n')
    fo.close()

def main(caffe_model):
  src_dir = './'
  caffe_proto = os.path.join(src_dir, 'caffe.proto')
  dst_dir = './'

  if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)
  os.system('touch {}/__init__.py'.format(dst_dir))
  compile_caffe(src_dir, dst_dir, caffe_proto)

  #caffemodel = 'train_iter_7750.caffemodel'
  caffemodel = caffe_model
  read_caffemodel(caffemodel)


if __name__ == '__main__':
    if not len(sys.argv) == 1:
        main(str(sys.argv[1]))
    else:
        main("deploy.caffemodel")

