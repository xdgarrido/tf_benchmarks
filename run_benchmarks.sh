#!/bin/bash

VENDOR=AMD
PRECISION=fp32
ITER1=5000
ITER2=10000
ITER3=1000

# Bert encoder Benchmark
# --length  (sequence length) 
# --size  (size of attention head) 
# --heads (number of heads)
# --layers (number of layers)
# --batch batch size
echo "Executing Bert Encoder !"
./BertEncoder/scripts/run_enc.sh --iter=$ITER1 --length=64 --batch=4 --heads=16 --layers=4  --precision=$PRECISION --vendor=$VENDOR

# Attention benchmark
# --length  (sequence length) 
# --size  (size of attention head) 
# --heads (number of heads)
# --batch batch size
echo "Executing Attention !"
./Attention/scripts/run_attention.sh --iter=$ITER1 --length=128 --batch=8  --size=64 --heads=16 --precision=$PRECISION --vendor=$VENDOR

# Embeddings Benchmark
# --length  (sequence length) 
# --size  (size of attention head) 
# --heads (number of heads)
# --layers (number of layers)
# --batch batch size
echo "Executing Embeddings (input processing) !"
./Embeddings/scripts/run_emb.sh --iter=$ITER1 --length=64 --batch=4 --heads=16 --layers=4  --precision=$PRECISION --vendor=$VENDOR

# Average pool layer 
# Allows strides and pooling blocks
echo "Executing AvgpoolLayer!"
./AvgpoolLayer/scripts/run_avglayer.sh --iter=$ITER3 --batch=16  --width=32 --height=32 --channels=3 --stride=1 --pool=2  --precision=$PRECISION --vendor=$VENDOR
./AvgpoolLayer/scripts/run_avglayer.sh --iter=$ITER3 --batch=16  --width=32 --height=32 --channels=3 --stride=2 --pool=2  --precision=$PRECISION --vendor=$VENDOR

# Max pool layer 
# Allows strides and pooling blocks
echo "Executing MaxpoolLayer!"
./MaxpoolLayer/scripts/run_maxlayer.sh --iter=$ITER3 --batch=16  --width=32 --height=32 --channels=3 --stride=1 --pool=2  --precision=$PRECISION --vendor=$VENDOR
./MaxpoolLayer/scripts/run_maxlayer.sh --iter=$ITER3 --batch=16  --width=32 --height=32 --channels=3 --stride=2 --pool=2  --precision=$PRECISION --vendor=$VENDOR

# Batch Layer Normalization
# --size (hiidden size)
echo "Executing BatchNormLayer!"
./BatchNormLayer/scripts/run_batchnorm.sh  --iter=$ITER2 --batch=64 --size=1024  --precision=$PRECISION --vendor=$VENDOR

# Layer Normalization
# --size (hiidden size)
echo "Executing Layer Normalization!"
./NormLayer/scripts/run_layernorm.sh  --iter=$ITER2 --batch=64 --size=1024  --precision=$PRECISION --vendor=$VENDOR

# Dropout Layer 
# --heads (number of heads)
#
echo "Executing Dropout Layer!"
./DropoutLayer/scripts/run_dropout.sh --iter=$ITER2 --length=128 --batch=8 --heads=16 --precision=$PRECISION --vendor=$VENDOR

# Convolution Layer
#  --activation=$ACTIVATION (relu,sigmoid,softmax,softplus,softsign,tanh,selu,elu,exponential)
#   image width, image height and  channels (rgb)
#   kernel dilation, strides on the source data
echo "Executing Convolutional Layer!"
./Conv2Layer/scripts/run_conv2layer.sh --iter=$ITER3 --batch=16  --width=32 --height=32 --channels=3 --kernel=3 --stride=1 --dilation=1 --activation=relu --precision=$PRECISION --vendor=$VENDOR
./Conv2Layer/scripts/run_conv2layer.sh --iter=$ITER3 --batch=16  --width=32 --height=32 --channels=3 --kernel=3 --stride=2 --dilation=1 --activation=relu --precision=$PRECISION --vendor=$VENDOR
./Conv2Layer/scripts/run_conv2layer.sh --iter=$ITER3 --batch=16  --width=32 --height=32 --channels=3 --kernel=3 --stride=1 --dilation=2 --activation=relu --precision=$PRECISION --vendor=$VENDOR
./Conv2Layer/scripts/run_conv2layer.sh --iter=$ITER3 --batch=16  --width=32 --height=32 --channels=3 --kernel=5 --stride=1 --dilation=1 --activation=relu --precision=$PRECISION --vendor=$VENDOR
./Conv2Layer/scripts/run_conv2layer.sh --iter=$ITER3 --batch=16  --width=32 --height=32 --channels=3 --kernel=3 --stride=1 --dilation=1 --activation=softmax --precision=$PRECISION --vendor=$VENDOR
./Conv2Layer/scripts/run_conv2layer.sh --iter=$ITER3 --batch=16  --width=32 --height=32 --channels=3 --kernel=3 --stride=2 --dilation=1 --activation=softmax --precision=$PRECISION --vendor=$VENDOR
./Conv2Layer/scripts/run_conv2layer.sh --iter=$ITER3 --batch=16  --width=32 --height=32 --channels=3 --kernel=3 --stride=1 --dilation=2 --activation=softmax --precision=$PRECISION --vendor=$VENDOR
./Conv2Layer/scripts/run_conv2layer.sh --iter=$ITER3 --batch=16  --width=32 --height=32 --channels=3 --kernel=5 --stride=1 --dilation=1 --activation=softmax --precision=$PRECISION --vendor=$VENDOR
# Dense Layer
#  --activation=$ACTIVATION (relu,sigmoid,softmax,softplus,softsign,tanh,selu,elu,exponential)
#  --size (hiidden size)
echo "Executing Dense Layer!"
./DenseLayer/scripts/run_layerdense.sh  --iter=$ITER2 --size=1024  --activation=relu --batch=16 --precision=$PRECISION --vendor=$VENDOR
./DenseLayer/scripts/run_layerdense.sh  --iter=$ITER2 --size=1024  --activation=softmax --batch=16 --precision=$PRECISION --vendor=$VENDOR

# Optimizers 
# --optimizer (adam, lamb, nadam or nlamb)"
echo "Executing Optimizers!"
./Optimizers/scripts/run_optimizer.sh --iter=$ITER2 --netsize=1000000 --optimizer=adam --vendor=$VENDOR
./Optimizers/scripts/run_optimizer.sh --iter=$ITER2 --netsize=1000000 --optimizer=lamb --vendor=$VENDOR
./Optimizers/scripts/run_optimizer.sh --iter=$ITER2 --netsize=1000000 --optimizer=nadam --vendor=$VENDOR
./Optimizers/scripts/run_optimizer.sh --iter=$ITER2 --netsize=1000000 --optimizer=nlamb --vendor=$VENDOR
cat eval_results.txt << EOF
