from net_compiler import *
from caffe_compiler import *

if __name__ == "__main__":
    if len(sys.argv) != 3:
        python_str = "python"
        this_pyfile_path = sys.argv[0]
        print("Usage: %s %s CAFFE_PROTOTXT CAFFE_CAFFEMODEL\n" % (python_str, this_pyfile_path))
        exit(-1)
    else:
        # net_compiler
        prototxt_file = sys.argv[1]
        net = Net(prototxt_file)
        name_pattern = "(.*).prototxt"
        name = re.findall(name_pattern, prototxt_file)[0]
        print("[INFO] Successful conversion from {}.prototxt to {}.c and {}.h" \
              .format(name, name, name))

        # caffe_compiler
        print("[INFO] Start to convert weight file ......")
        caffe_model_path = sys.argv[2]
        main(caffe_model_path)
        print("[INFO] Successful conversion from {}.caffemodel to {}.dat" \
              .format(name, name))
