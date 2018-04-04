import sys
import caffe
import numpy as np
from collections import OrderedDict
from cfg import *
from prototxt import *

DEBUG = True
log_handler = open('darknet2caffe_convert.log', 'w')
sys.stdout = log_handler

def darknet2caffe(cfgfile, weightfile, protofile, caffemodel):
    net_info = cfg2prototxt(cfgfile)
    save_prototxt(net_info , protofile, region=False)

    net = caffe.Net(protofile, caffe.TEST)
    params = net.params

    blocks = parse_cfg(cfgfile)
    fp = open(weightfile, 'rb')
    header = np.fromfile(fp, count=4, dtype=np.int32)
    buf = np.fromfile(fp, dtype = np.float32)
    fp.close()

    layers = []
    layer_id = 1
    start = 0
    for block in blocks:
        if start >= buf.size:
            break
        print('%s ' % layer_id + block['type'])
        if block['type'] == 'net':
            continue
        elif block['type'] == 'convolutional':
            batch_normalize = int(block['batch_normalize'])
            if block.has_key('name'):
                conv_layer_name = block['name']
                bn_layer_name = '%d_bn' % block['name']
                scale_layer_name = '%d_scale' % block['name']
            else:
                conv_layer_name = 'layer%d_conv' % layer_id
                bn_layer_name = 'layer%d_bn' % layer_id
                scale_layer_name = 'layer%d_scale' % layer_id

            if batch_normalize:
                start = load_conv_bn2caffe(buf, start, params[conv_layer_name], params[bn_layer_name], params[scale_layer_name])
            else:
                start = load_conv2caffe(buf, start, params[conv_layer_name])
            layer_id = layer_id+1
        elif block['type'] == 'connected':
            if block.has_key('name'):
                fc_layer_name = block['name']
            else:
                fc_layer_name = 'layer%d_fc' % layer_id
            start = load_fc2caffe(buf, start, params[fc_layer_name])
            layer_id = layer_id + 1
        elif block['type'] == 'maxpool':
            layer_id = layer_id + 1
        elif block['type'] == 'avgpool':
            layer_id = layer_id + 1
        elif block['type'] == 'route':
            layer_id = layer_id + 1
        elif block['type'] == 'shortcut':
            layer_id = layer_id + 1
        elif block['type'] == 'softmax':
            layer_id = layer_id + 1
        elif block['type'] == 'cost':
            layer_id = layer_id + 1
        elif block['type'] == 'reorg':
            layer_id = layer_id + 1
        else:
            print("============== unknow ==============")
            print('unknow layer type %s ' % block['type'])
            layer_id = layer_id + 1
    print('save prototxt to %s' % protofile)
    save_prototxt(net_info , protofile, region=True)
    print('save caffemodel to %s' % caffemodel)
    net.save(caffemodel)

def load_conv2caffe(buf, start, conv_param):
    weight = conv_param[0].data
    bias = conv_param[1].data
    conv_param[1].data[...] = np.reshape(buf[start:start+bias.size], bias.shape);   start = start + bias.size
    conv_param[0].data[...] = np.reshape(buf[start:start+weight.size], weight.shape); start = start + weight.size
    return start

def load_fc2caffe(buf, start, fc_param):
    weight = fc_param[0].data
    bias = fc_param[1].data
    fc_param[1].data[...] = np.reshape(buf[start:start+bias.size], bias.shape);   start = start + bias.size
    fc_param[0].data[...] = np.reshape(buf[start:start+weight.size], weight.shape); start = start + weight.size
    return start

def load_conv_bn2caffe(buf, start, conv_param, bn_param, scale_param):
    conv_weight = conv_param[0].data
    running_mean = bn_param[0].data
    running_var = bn_param[1].data
    scale_weight = scale_param[0].data
    scale_bias = scale_param[1].data

    scale_param[1].data[...] = np.reshape(buf[start:start+scale_bias.size], scale_bias.shape); start = start + scale_bias.size
    scale_param[0].data[...] = np.reshape(buf[start:start+scale_weight.size], scale_weight.shape); start = start + scale_weight.size
    bn_param[0].data[...] = np.reshape(buf[start:start+running_mean.size], running_mean.shape); start = start + running_mean.size
    bn_param[1].data[...] = np.reshape(buf[start:start+running_var.size], running_var.shape); start = start + running_var.size
    bn_param[2].data[...] = np.array([1.0])
    conv_param[0].data[...] = np.reshape(buf[start:start+conv_weight.size], conv_weight.shape); start = start + conv_weight.size
    return start

def cfg2prototxt(cfgfile):
    blocks = parse_cfg(cfgfile)

    layers = []
    props = OrderedDict() 
    bottom = 'data'
    layer_id = 1
    topnames = dict()

    for bidx in xrange(len(blocks)):
        block = blocks[bidx]
        if block['type'] == 'net':
            props['name'] = 'Darkent2Caffe'
            props['input'] = 'data'
            #props['input_dim'] = [block['batch']] #['1']
            props['input_dim'] = ['1']
            props['input_dim'].append(block['channels'])
            props['input_dim'].append(block['height'])
            props['input_dim'].append(block['width'])
            continue
        elif block['type'] == 'convolutional':
            conv_layer = OrderedDict()
            if block.has_key('name'):
                conv_layer['name'] = block['name']
                conv_layer['type'] = 'Convolution'
                conv_layer['bottom'] = bottom
                conv_layer['top'] = block['name']
            else:
                conv_layer['name'] = 'layer%d_conv' % layer_id
                conv_layer['type'] = 'Convolution'
                conv_layer['bottom'] = bottom
                conv_layer['top'] = 'layer%d_conv' % layer_id
            convolution_param = OrderedDict()
            convolution_param['num_output'] = block['filters']
            convolution_param['kernel_size'] = block['size']
            if block['pad'] == '1':
                convolution_param['pad'] = str(int(convolution_param['kernel_size'])/2)
            convolution_param['stride'] = block['stride']
            if block['batch_normalize'] == '1':
                convolution_param['bias_term'] = 'false'
            else:
                convolution_param['bias_term'] = 'true'
            conv_layer['convolution_param'] = convolution_param
            layers.append(conv_layer)
            bottom = conv_layer['top']

            if block['batch_normalize'] == '1':
                bn_layer = OrderedDict()
                if block.has_key('name'):
                    bn_layer['name'] = '%d_bn' % block['name']
                else:
                    bn_layer['name'] = 'layer%d_bn' % layer_id
                bn_layer['type'] = 'BatchNorm'
                bn_layer['bottom'] = bottom
                bn_layer['top'] = bottom
                batch_norm_param = OrderedDict()
                batch_norm_param['use_global_stats'] = 'true'
                bn_layer['batch_norm_param'] = batch_norm_param
                layers.append(bn_layer)

                scale_layer = OrderedDict()
                if block.has_key('name'):
                    scale_layer['name'] = '%d_scale' % block['name']
                else:
                    scale_layer['name'] = 'layer%d_scale' % layer_id
                scale_layer['type'] = 'Scale'
                scale_layer['bottom'] = bottom
                scale_layer['top'] = bottom
                scale_param = OrderedDict()
                scale_param['bias_term'] = 'true'
                scale_layer['scale_param'] = scale_param
                layers.append(scale_layer)

            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                if block.has_key('name'):
                    relu_layer['name'] = '%d_act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d_act' % layer_id
                relu_layer['type'] = 'ReLU'
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        elif block['type'] == 'maxpool':
            max_layer = OrderedDict()
            if block.has_key('name'):
                max_layer['name'] = block['name']
                max_layer['type'] = 'Pooling'
                max_layer['bottom'] = bottom
                max_layer['top'] = block['name']
            else:
                max_layer['name'] = 'layer%d_maxpool' % layer_id
                max_layer['type'] = 'Pooling'
                max_layer['bottom'] = bottom
                max_layer['top'] = 'layer%d_maxpool' % layer_id
            pooling_param = OrderedDict()
            pooling_param['kernel_size'] = block['size']
            pooling_param['stride'] = block['stride']

            # [special case] for stride=1 kernel_size=2 maxpool
            #                change kernel_size=2 to kernel_size=1
            #                after this convertor
            #                change back from kernel_size=1 to kernel_size=2
            if pooling_param['kernel_size'] == "2" and \
               pooling_param['stride'] == '1':
                print("blocks{} is a special pooling, stride={}, kernel_size={}" \
                      .format(bidx, \
                              pooling_param['stride'], \
                              pooling_param['kernel_size']))
                pooling_param['kernel_size'] = "1"

            pooling_param['pool'] = 'MAX'
            if block.has_key('pad') and int(block['pad']) == 1:
                pooling_param['pad'] = str((int(block['size'])-1)/2)
            max_layer['pooling_param'] = pooling_param
            layers.append(max_layer)
            bottom = max_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        elif block['type'] == 'avgpool':
            avg_layer = OrderedDict()
            if block.has_key('name'):
               avg_layer['name'] = block['name']
               avg_layer['type'] = 'Pooling'
               avg_layer['bottom'] = bottom
               avg_layer['top'] = block['name']
            else:
               avg_layer['name'] = 'layer%d_avgpool' % layer_id
               avg_layer['type'] = 'Pooling'
               avg_layer['bottom'] = bottom
               avg_layer['top'] = 'layer%d_avgpool' % layer_id
            pooling_param = OrderedDict()
            pooling_param['kernel_size'] = 7
            pooling_param['stride'] = 1
            pooling_param['pool'] = 'AVE'
            avg_layer['pooling_param'] = pooling_param
            layers.append(avg_layer)
            bottom = avg_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        elif block['type'] == 'region':
            region_layer = OrderedDict()
            if block.has_key('name'):
                region_layer['name'] = block['name']
                region_layer['type'] = 'Region'
                region_layer['bottom'] = bottom
                region_layer['top'] = block['name']
            else:
                region_layer['name'] = 'layer%d_region' % layer_id
                region_layer['type'] = 'Region'
                region_layer['bottom'] = bottom
                region_layer['top'] = 'layer%d_region' % layer_id
                region_param = OrderedDict()
                region_param['anchors'] = block['anchors'].strip()
                region_param['classes'] = block['classes']
                region_param['bias_match'] = block['bias_match']
                region_param['coords'] = block['coords']
                region_param['num'] = block['num']
                region_param['softmax'] = block['softmax']
                region_param['jitter'] = block['jitter']
                region_param['rescore'] = block['rescore']

                region_param['object_scale'] = block['object_scale']
                region_param['noobject_scale'] = block['noobject_scale']
                region_param['class_scale'] = block['class_scale']
                region_param['coord_scale'] = block['coord_scale']

                region_param['absolute'] = block['absolute']
                region_param['thresh'] = block['thresh']
                region_param['random'] = block['random']

                # other hyper-parameters not int cfg file
                region_param['nms_thresh'] = 0.3
                region_param['background'] = 0
                region_param['tree_thresh'] = 0.5
                region_param['relative'] = 1
                region_param['box_thresh'] = 0.24


            region_layer['region_param'] = region_param
            layers.append(region_layer)
            bottom = region_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id + 1
        elif block['type'] == 'route':
            print("block:%s" % block)       
            from_layers = block['layers'].split(',')
            if len(from_layers) == 1:
                prev_layer_id = layer_id + int(from_layers[0])
                bottom = topnames[prev_layer_id]
                topnames[layer_id] = bottom
                layer_id = layer_id + 1
            else:
                prev_layer_id1 = layer_id + int(from_layers[0]) 
                prev_layer_id2 = layer_id + int(from_layers[1])
                print("from_layer: %s" % from_layers)
                print("prev_layer_id1: %s" % prev_layer_id1)
                print("prev_layer_id2: %s" % prev_layer_id2)
                print("layer_id: %s" % layer_id)

                bottom1 = topnames[prev_layer_id1]
                bottom2 = topnames[prev_layer_id2]
                concat_layer = OrderedDict()
                if block.has_key('name'):
                    concat_layer['name'] = block['name']
                    concat_layer['type'] = 'Concat'
                    concat_layer['bottom'] = [bottom1, bottom2]
                    concat_layer['top'] = block['name']
                else:
                    concat_layer['name'] = 'layer%d_concat' % layer_id
                    concat_layer['type'] = 'Concat'
                    concat_layer['bottom'] = [bottom1, bottom2]
                    concat_layer['top'] = 'layer%d_concat' % layer_id
                print("concat_layer: %s" % concat_layer)
                layers.append(concat_layer)
                bottom = concat_layer['top']
                topnames[layer_id] = bottom
                layer_id = layer_id+1
        elif block['type'] == 'shortcut':
            prev_layer_id1 = layer_id + int(block['from'])
            prev_layer_id2 = layer_id - 1
            bottom1 = topnames[prev_layer_id1]
            bottom2= topnames[prev_layer_id2]
            shortcut_layer = OrderedDict()
            if block.has_key('name'):
                shortcut_layer['name'] = block['name']
                shortcut_layer['type'] = 'Eltwise'
                shortcut_layer['bottom'] = [bottom1, bottom2]
                shortcut_layer['top'] = block['name']
            else:
                shortcut_layer['name'] = 'layer%d_shortcut' % layer_id
                shortcut_layer['type'] = 'Eltwise'
                shortcut_layer['bottom'] = [bottom1, bottom2]
                shortcut_layer['top'] = 'layer%d_shortcut' % layer_id
            eltwise_param = OrderedDict()
            eltwise_param['operation'] = 'SUM'
            shortcut_layer['eltwise_param'] = eltwise_param
            layers.append(shortcut_layer)
            bottom = shortcut_layer['top']
 
            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                if block.has_key('name'):
                    relu_layer['name'] = '%d_act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d_act' % layer_id
                relu_layer['type'] = 'ReLU'
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1           
            
        elif block['type'] == 'connected':
            fc_layer = OrderedDict()
            if block.has_key('name'):
                fc_layer['name'] = block['name']
                fc_layer['type'] = 'InnerProduct'
                fc_layer['bottom'] = bottom
                fc_layer['top'] = block['name']
            else:
                fc_layer['name'] = 'layer%d_fc' % layer_id
                fc_layer['type'] = 'InnerProduct'
                fc_layer['bottom'] = bottom
                fc_layer['top'] = 'layer%d_fc' % layer_id
            fc_param = OrderedDict()
            fc_param['num_output'] = int(block['output'])
            fc_layer['inner_product_param'] = fc_param
            layers.append(fc_layer)
            bottom = fc_layer['top']

            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                if block.has_key('name'):
                    relu_layer['name'] = '%d_act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d_act' % layer_id
                relu_layer['type'] = 'ReLU'
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        elif block['type'] == 'reorg':
            print("block reorg bidx: %d" % bidx)
            reshape_layer = OrderedDict()
            if block.has_key('name'):
                avg_layer['name'] = block['name']
                reshape_layer['type'] = 'Reshape'
                #reshape_layer['type'] = 'Reorg'
                reshape_layer['bottom'] = bottom
                avg_layer['top'] = block['name']
            else:
                reshape_layer['name'] = 'layer%d_reorg' % layer_id
                reshape_layer['type'] = 'Reshape'
                #reshape_layer['type'] = 'Reorg'
                reshape_layer['bottom'] = bottom
                reshape_layer['top'] = 'layer%d_reorg' % layer_id
            reshape_param = OrderedDict()
            shape = OrderedDict()
            # TODO CHECK YOLOv2
            # step1: find fisrt block['type'] == 'route'
            print("[step1] find first block['type'] == 'route'")
            block_n_idx_tuple_list = map(lambda b, idx: \
                                            (b['type'], idx), \
                                          blocks, xrange(len(blocks)))
            route_tuple_list = filter(lambda t: \
                                         t[0] == "route",\
                                      block_n_idx_tuple_list)
            first_route_idx = route_tuple_list[0][1]
            first_route_block = blocks[first_route_idx]
            if first_route_block['type'] == "route":
                first_route_from_layers_list = first_route_block['layers'].split(',')
                print("first_route_from_layers_list: " + str(first_route_from_layers_list))
                if len(first_route_from_layers_list) == 1:
                    # start from 1 and 1 is net, not first conv, thus minus 2
                    first_route_from_layers_idx = first_route_idx + int(first_route_from_layers_list[0]) - 0
                    print("blocks[first_route_from_layers_idx]['type']: %s" % blocks[first_route_from_layers_idx]['type'])
                    print("first_route_from_layers_idx:%d" % first_route_from_layers_idx)
                    print("blocks[first_route_from_layers_idx]:%s" % str(blocks[first_route_from_layers_idx]))
                    #print("blocks[14]:%s" % str(blocks[14]))

                    # step2: store stride from begin to prev_layer_id in stride_list
                    print("[step2] store stride from begin to prev_layer_id in stride_list")
                    stride_list = []
                    reorg_input_filter_num = 0
                    second_route_idx = route_tuple_list[1][1]
                    second_route_block = blocks[second_route_idx]
                    print("second_route_idx:%s" % str(second_route_idx))
                    second_route_from_layers_list = second_route_block['layers'].split(',')
                    print("second_route_from_layers_list:%s" % str(second_route_from_layers_list))
                    second_route_from_layers_idx_list = map(lambda ind_str: 
                                                                   second_route_idx + int(ind_str) - 0,
                                                            second_route_from_layers_list)
                    print("second_route_from_layers_idx_list:%s" % str(second_route_from_layers_idx_list))
                    #for bbidx in xrange(len(blocks[:prev_layer_id+1])):
                    #                                     right=13(second_route[0])======reorg bidx
                    # ====master branch===9(last, not first_route, but first_route[0])
                    # TODO                                left=10(second_route[1])=======
                    master_branch_end_bidx = first_route_from_layers_idx
                    right_branch_only_start_bidx = first_route_idx
                    reorg_bidx = bidx
                    right_branch_only_end_bidx = reorg_bidx
                    print("\n========== key bidx check =========")
                    print("reorg bidx:%s" % bidx)
                    print("first route idx:%s" % first_route_idx)
                    print("second route idx:%s" % second_route_idx)
                    print("master_branch_end_bidx: %s" % master_branch_end_bidx)
                    print("str(blocks[master_branch_end_bidx]): %s" % str(blocks[master_branch_end_bidx]))

                    print
                    print("right_branch_only_start_bidx:%s" % right_branch_only_start_bidx)
                    print("str(blocks[right_branch_only_start_bidx]): %s" % str(blocks[right_branch_only_start_bidx]))

                    print
                    print("right_branch_only_end_bidx: %s" % right_branch_only_end_bidx)
                    print("str(blocks[right_branch_only_end_bidx]): %s" % str(blocks[right_branch_only_end_bidx]))
                    print("=====================================\n")

                    # include last block, so +1 for python index
                    master_branch_blocks = blocks[:master_branch_end_bidx+1]
                    right_branch_blocks_only_before_reorg = blocks[right_branch_only_start_bidx:right_branch_only_end_bidx+1]
                    # check master branch and right branch
                    print("\n=============== check master branch =================")
                    for mbidx in xrange(len(master_branch_blocks)): 
                        master_block = master_branch_blocks[mbidx]
                        print("%d\t%s" % (mbidx, str(master_block)))

                    print("\n=============== check right branch ==================")
                    for rbidx in xrange(len(right_branch_blocks_only_before_reorg)):
                        right_block = right_branch_blocks_only_before_reorg[rbidx]
                        print("%d\t%s" % (rbidx, str(right_block)))

                    master_nd_right_branch_blocks = master_branch_blocks + right_branch_blocks_only_before_reorg
                    # Seach stride value for master and right branches
                    for mridx in xrange(len(master_nd_right_branch_blocks)):
                        mrblock = master_nd_right_branch_blocks[mridx]
                        print(mridx, mrblock['type'])
                        if mrblock['type'] == "convolutional" or \
                           mrblock['type'] == "maxpool" or \
                           mrblock['type'] == "avgpool":
                            stride = mrblock['stride']
                            stride_list.append(int(stride))

                        # input channels of reorg layer
                        if mrblock['type'] == "convolutional":
                            reorg_input_filter_num = int(mrblock['filters'])
                    print("stride_list:%s" % str(stride_list))
                    print("reorg_input_filter_num:%d" % reorg_input_filter_num)
                    # step3: compute input dimension of reorg layer
                    print("[step3] compute input dimension of reorg layer")
                    stride_factor = reduce(lambda a, b: a*b, stride_list)
                    print("stride_factor:%d" % stride_factor)
                    input_h = int(blocks[0]['height'])/stride_factor
                    input_w = int(blocks[0]['width'])/stride_factor

                    #batch_num = int(blocks[0]['batch'])
                    batch_num = 1
                    out_c = reorg_input_filter_num * int(block['stride'])**2
                    out_h = input_h / int(block['stride'])
                    out_w = input_w / int(block['stride'])
                    shape['dim'] = [batch_num, out_c, out_h, out_w]
                    print(shape['dim']) 
                else:
                    print("reorg layer error: first route block has more than one from-layers")
                    exit(-1)

            else:
                print("reorg layer error: former block of reorg block is not route block")
                exit(-1)
                
            reshape_param['shape'] = shape
            reshape_layer['reshape_param'] = reshape_param
            if DEBUG:
                for k in block:
                    print("block[%s]: %s" % (k, block[k]))
            layers.append(reshape_layer) 

            if DEBUG:
                print("========== reorg =========")
                print("reshape['top']: %s" % (reshape_layer['top']))
                print("layer_id: %s" % layer_id)
                print("bottom: %s" % bottom)
            bottom = reshape_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id + 1
        else:
            print('unknow layer type %s ' % block['type'])
            topnames[layer_id] = bottom
            layer_id = layer_id + 1

    net_info = OrderedDict()
    net_info['props'] = props
    net_info['layers'] = layers
    return net_info

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: python darknet2caffe.py DARKNET_CFG DARKNET_WEIGHTS')
        exit(-1)

    cfgfile = sys.argv[1]
    #net_info = cfg2prototxt(cfgfile)
    #print_prototxt(net_info)
    #save_prototxt(net_info, 'tmp.prototxt')
    weightfile = sys.argv[2]
    name = cfgfile.replace(".cfg", "")
    protofile = ".".join([name, "prototxt"])
    caffemodel = ".".join([name, "caffemodel"])

    darknet2caffe(cfgfile, weightfile, protofile, caffemodel)
    format_data_layer(protofile)
    correct_pooling_layer(cfgfile, protofile)
