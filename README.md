# darknet2inferx

This model convertor ported from [original](https://github.com/marvis/pytorch-caffe-darknet-convert) supports conversion from darkent to caffe, especially for YOLOv2 and tiny-YOLO etc. 

## Step1 Caffe Environment

First, ensure caffe installed (**converison progress'll use Python interface of caffe**), recommanding using [Docker image](https://hub.docker.com/r/bvlc/caffe/) of `bvlc/caffe:cpu` instead.

## Step2 Convert

Use following command, convert darknet model to caffe's:

```shell
python darknet2caffe.py DARKNET_CFG DARKNET_WEIGHTS
```

If last message shows as below, it means successful conversion from darknet to caffe:

```shell
Network initialization done.
```

Next is conversion from caffe to InferXLite:

```shell
python caffe2inferx.py CAFFE_PROTOTXT CAFFE_CAFFEMODEL
```

## Appendix

Translate to InferXLite directly from darknet:

```shell
python darknet2inferx.py DARKNET_CFG DARKNET_WEIGHTS
```

Check exectuion log in `darknet2caffe_convert.log`.

Translate `*.cfg` file to `*.prototxt` only:

```shell
python cfg.py DARKNET_CFG
```

## TODO

- [x] auto shape infer for output dimension of reorg layer from darknet to caffe, especially for **one-reorg-layer networks** like YOLOv2.
- [x] darknet2inferx
   - [x] support converison of region layer's parameters to variables in `*.h` file.
   - [ ] support `yolo_pooling` judge/choose in pooling conversion from caffe to inferxlite [DELAY]
- [ ] darknet2caffe
   - [ ] support conversion of **pooling layer** for special cases (stride=1 pooling, decimal value)
