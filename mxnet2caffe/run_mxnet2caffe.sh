model_root=/workspace/data/insightface-models/model-r18-spa-relu-m2.0-8gpu-v17_imdbface_above10_176513-128-H-V1/slim

python mxnet2caffe.py \
      --mx-model $model_root/model-r18-slim \
      --mx-epoch 145 \
      --cf-prototxt $model_root/caffe/model-r18-slim.prototxt \
      --cf-model $model_root/caffe/model-r18-slim-145.caffemodel \
 
