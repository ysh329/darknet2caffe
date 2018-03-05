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

### darknet2inferx

Translate to InferXLite directly from darknet:

```shell
python darknet2inferx.py DARKNET_CFG DARKNET_WEIGHTS
```

Check exectuion log in `darknet2caffe_convert.log`.

Translate `*.cfg` file to `*.prototxt` only:

```shell
python cfg.py DARKNET_CFG
```

### Old API

Due to the newest API starting with `inferx_` in `*.c` file (such as `inferx_convolution`), if use old API (without `inferx_`), you should convert to old API using command below:

```shell
python to_old_api_for_c_file.py INFERX_MODEL_C 
```


## TODO

- [x] auto shape infer for output dimension of reorg layer from darknet to caffe, especially for **one-reorg-layer networks** like YOLOv2.
- [x] darknet2inferx
   - [x] support converison of region layer's parameters to variables in `*.h` file.
   - [x] support `yolo_pooling` judge/choose in pooling conversion from caffe to inferxlite [DELAY]
- [x] darknet2caffe
   - [x] support conversion of **pooling layer** for a special case (input shape same as output shape. More concretely, stride=1 size=2 max pooling, this case's process of darknet will pad 1 for right and down side of input feature map. Thus, this conversion replaces `stride=1 size=2` with `stride=1 size=1` before `cfg2proto`. After conversion from `weights` to `caffemodel`, an afterward process'll replace pooling setting in cfg file using ground truth params (stride and size) in cfg file).
