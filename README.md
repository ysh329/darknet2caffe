# Dark2Caffe

This repository ported from [original](https://github.com/marvis/pytorch-caffe-darknet-convert) supports conversion from darkent to caffe, especially for YOLOv2 and tiny-YOLO etc. 

## Step1 Caffe Environment

First, ensure caffe installed, recommanding using Docker image of `bvlc/caffe:cpu` instead.

## Step2 Reorg Layer

After that, if your model is based on `YOLOv2` or having `reorg` layer (if not, you can ignore this step), you should define the output dimension of `reorg` layer in code `darknet2caffe.py` as below:

```python
            # TODO: auto shape infer
            shape['dim'] = [1, 2048, 9, 9]
```

If do not sure the output dimension of `reorg` layer, execute model again using darknet and check its execution log, which clearly shows the output dimension of `reorg` layer.

## Step3 Convert

After definination of `shape['dim']` variable, use command below to convert darknet model to caffe's:

```shell
python darknet2caffe.py DARKNET_CFG DARKNET_WEIGHTS CAFFE_PROTOTOXT CAFFE_CAFFEMODEL
```

If messages below shows, it means successful converison:

```shell
Network initialization done.
```

## TODO

- [ ] auto shape infer for output dimension of reorg layer
