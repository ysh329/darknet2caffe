from collections import OrderedDict
try:
    import caffe.proto.caffe_pb2 as caffe_pb2
except:
    try:
        import caffe_pb2
    except:
        print 'caffe_pb2.py not found. Try:'
        print '  protoc caffe.proto --python_out=.'
        exit()

def parse_caffemodel(caffemodel):
    model = caffe_pb2.NetParameter()
    print 'Loading caffemodel: ', caffemodel
    with open(caffemodel, 'rb') as fp:
        model.ParseFromString(fp.read())

    return model


def parse_prototxt(protofile):
    def line_type(line):
        if line.find(':') >= 0:
            return 0
        elif line.find('{') >= 0:
            return 1
        return -1

    def parse_block(fp):
        block = OrderedDict()
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0: # key: value
                #print line
                line = line.split('#')[0]
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                if block.has_key(key):
                    if type(block[key]) == list:
                        block[key].append(value)
                    else:
                        block[key] = [block[key], value]
                else:
                    block[key] = value
            elif ltype == 1: # blockname {
                key = line.split('{')[0].strip()
                sub_block = parse_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
            line = line.split('#')[0]
        return block

    fp = open(protofile, 'r')
    props = OrderedDict()
    layers = []
    line = fp.readline()
    while line != '':
        line = line.strip().split('#')[0]
        if line == '':
            line = fp.readline()
            continue
        ltype = line_type(line)
        if ltype == 0: # key: value
            key, value = line.split(':')
            key = key.strip()
            value = value.strip().strip('"')
            if props.has_key(key):
               if type(props[key]) == list:
                   props[key].append(value)
               else:
                   props[key] = [props[key], value]
            else:
                props[key] = value
        elif ltype == 1: # blockname {
            key = line.split('{')[0].strip()
            if key == 'layer':
                layer = parse_block(fp)
                layers.append(layer)
            else:
                props[key] = parse_block(fp)
        line = fp.readline()

    if len(layers) > 0:
        net_info = OrderedDict()
        net_info['props'] = props
        net_info['layers'] = layers
        return net_info
    else:
        return props

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def print_prototxt(net_info):
    # whether add double quote
    def format_value(value):
        #str = u'%s' % value
        #if str.isnumeric():
        if is_number(value):
            return value
        elif value == 'true' or value == 'false' or value == 'MAX' or value == 'SUM' or value == 'AVE':
            return value
        else:
            return '\"%s\"' % value

    def print_block(block_info, prefix, indent):
        blanks = ''.join([' ']*indent)
        print('%s%s {' % (blanks, prefix))
        for key,value in block_info.items():
            if type(value) == OrderedDict:
                print_block(value, key, indent+4)
            elif type(value) == list:
                for v in value:
                    print('%s    %s: %s' % (blanks, key, format_value(v)))
            else:
                print('%s    %s: %s' % (blanks, key, format_value(value)))
        print('%s}' % blanks)
        
    props = net_info['props']
    layers = net_info['layers']
    print('name: \"%s\"' % props['name'])
    print('input: \"%s\"' % props['input'])
    print('input_dim: %s' % props['input_dim'][0])
    print('input_dim: %s' % props['input_dim'][1])
    print('input_dim: %s' % props['input_dim'][2])
    print('input_dim: %s' % props['input_dim'][3])
    print('')
    for layer in layers:
        print_block(layer, 'layer', 0)

def save_prototxt(net_info, protofile, region=True):
    fp = open(protofile, 'w')
    # whether add double quote
    def format_value(value):
        #str = u'%s' % value
        #if str.isnumeric():
        if is_number(value):
            return value
        elif value == 'true' or value == 'false' or value == 'MAX' or value == 'SUM' or value == 'AVE':
            return value
        else:
            return '\"%s\"' % value

    def print_block(block_info, prefix, indent):
        blanks = ''.join([' ']*indent)
        print >>fp, '%s%s {' % (blanks, prefix)
        for key,value in block_info.items():
            if type(value) == OrderedDict:
                print_block(value, key, indent+4)
            elif type(value) == list:
                for v in value:
                    print >> fp, '%s    %s: %s' % (blanks, key, format_value(v))
            else:
                print >> fp, '%s    %s: %s' % (blanks, key, format_value(value))
        print >> fp, '%s}' % blanks
        
    props = net_info['props']
    layers = net_info['layers']
    """
    print >> fp, 'layer {'
    print >> fp, '  name: \"%s\"' % props['name']
    print >> fp, '  type: \"Input\"'
    print >> fp, '  top: \"data\"'
    print >> fp, '  input_param {'
    print >> fp, '    shape {'
    print >> fp, '      dim: %s' % props['input_dim'][0]
    print >> fp, '      dim: %s' % props['input_dim'][1]
    print >> fp, '      dim: %s' % props['input_dim'][2]
    print >> fp, '      dim: %s' % props['input_dim'][3]
    print >> fp, '    }'
    print >> fp, '}'
    """
    print >> fp, 'input: \"%s\"' % props['input']
    print >> fp, 'input_dim: %s' % props['input_dim'][0]
    print >> fp, 'input_dim: %s' % props['input_dim'][1]
    print >> fp, 'input_dim: %s' % props['input_dim'][2]
    print >> fp, 'input_dim: %s' % props['input_dim'][3]
    
    print >> fp, ''
    for layer in layers:
        if layer['type'] != 'Region' or region == True:
            print_block(layer, 'layer', 0)
    fp.close()



def correct_pooling_layer(cfgfile, protofile):
    # create a pool idx list for those pool with kernel_size=2 stride=1
    #     to record those sepcial pool's index in **all pool layers**
    #     this recorded pool index list is used to avoid conversion for original ksize=1,stride=1 pool
    #
    # special case pool: input shape same as output shape
    #     only for pool with stride=1 kernel_size=2
    #     during conversion from cfg to protofile
    #     kernel_size=2 changed to 1
    #     here kernel_size will change back to 2

    # init result
    with open(protofile, "r") as proto_handle:
        correct_proto_line_list = proto_handle.readlines()

    # step1. filter all pool in cfg file
    print("==== cfg ====")
    with open(cfgfile, "r") as cfg_handle:
        cfg_lines_str = cfg_handle.read()
        cfg_lines_str = cfg_lines_str.replace("\r", "")
        print("cfg_lines_str[:10]:%s" % cfg_lines_str[:10])

        import re
        pool_size_pattern = re.compile(r"\[.*pool\]\nsize=(\d)\n", re.M)
        pool_stride_pattern = re.compile(r"\[.*pool\]\nsize=\d\nstride=(\d)", re.M)
        pool_size_list = re.findall(pool_size_pattern, cfg_lines_str)
        pool_stride_list = re.findall(pool_stride_pattern, cfg_lines_str)
        print("pool_size_list:%s" % str(pool_size_list))
        print("pool_stride_list:%s" % str(pool_stride_list))
        # find pool
    pool_size_and_stride_tuple_list_cfg = map(lambda size, stride:
                                               (size, stride),
                                              pool_size_list, pool_stride_list)
    # step2. filter all pool in prototxt file
    print("==== prototxt ====")
    with open(protofile, "r") as proto_handle:
        proto_lines_str = proto_handle.read()
        proto_lines_str = proto_lines_str.replace("\r", "")
        print("proto_lines_str[:10]:%s" % proto_lines_str[:10])
        import re
        pool_kernel_size_pattern = re.compile(r"pooling_param {\n        kernel_size: (\d)\n", re.M)
        pool_stride_pattern = re.compile(r"pooling_param {\n        kernel_size: .*\n        stride: (\d)\n", re.M)
        pool_kernel_size_list = re.findall(pool_kernel_size_pattern, proto_lines_str)
        pool_stride_list = re.findall(pool_stride_pattern, proto_lines_str)
        print("pool_kernel_size_list:%s" % str(pool_kernel_size_list)) 
        print("pool_stride_list:%s" % str(pool_stride_list))
    pool_kernel_size_and_stride_tuple_list_proto = map(lambda kernel_size, stride:
                                                              (kernel_size, stride),
                                                       pool_kernel_size_list, pool_stride_list)
    # step3. compare pools between cfg and prototxt file
    if len(pool_kernel_size_and_stride_tuple_list_proto) == \
       len(pool_size_and_stride_tuple_list_cfg):

        for pool_idx in xrange(len(pool_size_and_stride_tuple_list_cfg)):
            cfg_pool = pool_size_and_stride_tuple_list_cfg[pool_idx]
            proto_pool = pool_kernel_size_and_stride_tuple_list_proto[pool_idx]

            # prototxt pool
            ksize_proto = proto_pool[0]
            stride_proto = proto_pool[1]
            # cfg pool
            ksize_cfg = cfg_pool[0]
            stride_cfg = cfg_pool[1]

            # compare pool from cfg and protofile
            if (ksize_proto == ksize_cfg) and \
               (stride_proto == stride_cfg):
                continue
                # same pool with same stride and kernel size
            else:
                print("==== replace ====")
                print("index %s's pool from cfg and proto are different" % str(pool_idx))
                print("cfg pool with stride=%s ksize=%s" % (stride_cfg, ksize_cfg))
                print("prototxt pool with stride=%s ksize=%s" % (stride_proto, ksize_proto))

                # step4. replace the target pool with cfg pool
                target_pool_idx = pool_idx
                with open(protofile, "r") as proto_handle:
                    proto_line_list = proto_handle.readlines()
                    idx_and_line_tuple_list = map(lambda line_idx, line: \
                                                         (line_idx, line), \
                                                  xrange(len(proto_line_list)), proto_line_list)
                    print("idx_and_line_tuple_list[0]:%s" % str(idx_and_line_tuple_list[0]))
                    print("type(idx_and_line_tuple_list):%s" % type(idx_and_line_tuple_list))
                    poolStartIdx_and_line_tuple_list = filter(lambda (line_idx, line): \
                                                                     'pooling_param' in line, \
                                                                     idx_and_line_tuple_list)
                    print("poolStartIdx_and_line_tuple_list:%s" % str(poolStartIdx_and_line_tuple_list))
                    # target pool line found
                    target_pool_line_idx = int(poolStartIdx_and_line_tuple_list[target_pool_idx][0])
                    print("target_pool_line_idx:%d" % target_pool_line_idx)
                    target_ksize_and_stride_str = "".join([proto_line_list[target_pool_line_idx+1],\
                                                           proto_line_list[target_pool_line_idx+2]])
                    print("proto_line_list[target_pool_line_idx+1]:%s" % proto_line_list[target_pool_line_idx+1])
                    print("proto_line_list[target_pool_line_idx+2]:%s" % proto_line_list[target_pool_line_idx+2])
                    ksize_pattern = r".*kernel_size: (\d)\n"
                    stride_pattern = r".*stride: (\d)\n"
                    try:
                        import re
                        ksize = re.findall(ksize_pattern, target_ksize_and_stride_str)[0]
                        stride = re.findall(stride_pattern, target_ksize_and_stride_str)[0]
                        print("==== proto before replace ====")
                        print("ksize:%s" % ksize)
                        print("stride:%s" % stride)
                    except:
                        print("[ERROR] no ksize or stride param(s) found in prototxt")
                        exit(-1)
                    # replace
                    ksize_proto_line_str = "        kernel_size: {}\n".format(ksize_cfg)
                    stride_proto_line_str = "        stride: {}\n".format(stride_cfg)
                    print("==== check replace ====")
                    print("ksize_proto_line_str:%s" % ksize_proto_line_str)
                    print("stride_proto_line_str:%s" % stride_proto_line_str)
                    print("len(correct_proto_line_list)):%s" % len(correct_proto_line_list))
                    # special pad for inferxlite's pooling_yolo layer
                    #     only for this special stride=1 pooling
                    pad_proto_line_str = "        pad: {}\n".format("1")
                    correct_proto_line_list[target_pool_line_idx+1] = ksize_proto_line_str
                    correct_proto_line_list[target_pool_line_idx+2] = "".join([stride_proto_line_str, pad_proto_line_str])
        print("==== finish ====")
    else:
        print("[ERROR] the number of pools from cfg differs from the number of the one from prototxt")
        exit(-1)
    # step5. save prototxt result
    with open(protofile, "w") as proto_handle:
        proto_handle.writelines(correct_proto_line_list)
        

def format_data_layer(protofile):
    model_name_pattern = '(.*)\..*'
    dim_pattern = 'input_dim: (.*)'
    with open(protofile) as protofile_handle:
        lines = protofile_handle.readlines()
   
    try:
        import re
        # model name
        proto_name = re.findall(model_name_pattern, protofile)[0]
        split_list = map(lambda char, idx: (char == "/", idx), proto_name, xrange(len(proto_name)))
        split_list = filter(lambda (is_split, idx): is_split == True, split_list)
        if len(split_list) >= 1:
            split_idx = split_list[-1][1]
            model_name = proto_name[split_idx+1:]
        else:
            model_name = proto_name

        dim = [re.findall(dim_pattern, lines[1])[0],
               re.findall(dim_pattern, lines[2])[0],
               re.findall(dim_pattern, lines[3])[0],
               re.findall(dim_pattern, lines[4])[0],
              ]
    except:
        print("Don't need to format data layer")
        savefile_handle.close()
        return

    dim = map(str, dim)
    data_layer_str = '''name: "%(model_name)s"

layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: %(dim0)s
      dim: %(dim1)s
      dim: %(dim2)s
      dim: %(dim3)s
    }    
  }
}\n''' % {'model_name': model_name, 'dim0': dim[0], 'dim1': dim[1], 'dim2': dim[2], 'dim3': dim[3]}

    print(data_layer_str)

    proto_lines_str = data_layer_str + reduce(lambda l1,l2: l1+l2, lines[5:])

    savefile_handle = open(protofile, "w")
    savefile_handle.write(proto_lines_str)    
    savefile_handle.close()
        

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: python prototxt.py model.prototxt')
        exit()

    net_info = parse_prototxt(sys.argv[1])
    print_prototxt(net_info)
    save_prototxt(net_info, 'tmp.prototxt')
