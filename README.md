# darknet2inferx

This model convertor ported from [original](https://github.com/marvis/pytorch-caffe-darknet-convert) supports conversion from darkent to caffe, especially for YOLOv2 and tiny-YOLO etc. 

## Step1 Caffe Environment

First, ensure caffe installed (**converison progress'll use Python interface of caffe**), recommanding using [Docker image](https://hub.docker.com/r/bvlc/caffe/) of `bvlc/caffe:cpu` instead.

## Step2 Convert

Use following command, convert darknet model to caffe's:

```shell
python darknet2caffe.py DARKNET_CFG DARKNET_WEIGHTS CAFFE_PROTOTXT CAFFE_CAFFEMODEL
```

If last message shows as below, it means successful conversion:

```shell
Network initialization done.
```

## TODO

- [x] auto shape infer for output dimension of reorg layer, especially for **one-reorg-layer networks** like YOLOv2.
- [x] darknet2inferx
   - [x] support converison of region layer's parameters to variables in `*.h` file.
   - [ ] support `yolo_pooling` judge/choose in pooling conversion[DELAY]
