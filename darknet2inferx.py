from darknet2caffe import *
from caffe2inferx import *

if __name__ == "__main__":
    import sys 
    if len(sys.argv) != 3:
        print('Usage: python {} DARKNET_CFG DARKNET_WEIGHTS'.format(sys.argv[0]))
        exit(-1)

    ############################################################
    # Step1. convert to Caffe's *.prototxt, *.caffemodel files #
    #        from Darknet                                      #
    ############################################################
    cfg_file = sys.argv[1]
    weights_file = sys.argv[2]
    name = cfg_file.replace(".cfg", "")
    prototxt_file = ".".join([name, "prototxt"])
    caffemodel_file = ".".join([name, "caffemodel"])

    darknet2caffe(cfg_file, weights_file, prototxt_file, caffemodel_file)
    format_data_layer(prototxt_file)
    correct_pooling_layer(cfg_file, prototxt_file)
    #########################################################
    # Step2. convert to InferXLite's *.c, *.h, *.dat files  #
    #        from Caffe                                     #
    #########################################################
    # net_compiler
    net = Net(prototxt_file)
    print("[INFO] Successful conversion from {}.prototxt to {}.c and {}.h" \
          .format(name, name, name))

    # caffe_compiler
    print("[INFO] Start to convert weight file ......")
    main(caffemodel_file)
    print("[INFO] Successful conversion from {}.caffemodel to {}.dat" \
          .format(name, name))

