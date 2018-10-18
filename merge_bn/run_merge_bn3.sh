model_root=/workspace/data/insightface-models/model-r18-spa-relu-m2.0-8gpu-v17_imdbface_above10_176513-128-H-V1/slim

python merge_bn3.py \
    --model=$model_root/caffe/model-r18-slim.prototxt \
    --weights=$model_root/caffe/model-r18-slim-145.caffemodel \
    --merged-model=$model_root/caffe-merge-bn/model-r18-slim-merge-bn.prototxt \
    --merged-weights=$model_root/caffe-merge-bn/model-r18-slim-145-merge-bn.caffemodel 
