export PATH=$PATH:../bin/
function usage()
{
    echo "Usage:"
    echo ""
    echo "./run_conv2layer.sh"
    echo "\t-h --help"
    echo "\t--iter=$COUNT (number of iterations) "
    echo "\t--batch=$BATCH (batch_size) "
    echo "\t--activation=$ACTIVATION (relu,sigmoid,softmax,softplus,softsign,tanh,selu,elu,exponential) "
    echo "\t--width=$WIDTH (image width) "
    echo "\t--height=$HEIGHT (image height) "
    echo "\t--channels=$CHANNELS (image channels) "
    echo "\t--kernel=$KERNEL (kernel size)"
    echo "\t--stride=$STRIDE (conv. stride)"
    echo "\t--dilation=$DILATION (conv. dilation)"
    echo "\t--mode=$MODE (benchmark or validation)"
    echo "\t--precision=$PRECISION (fp32 or fp16)"
}

secs_to_human() {
    if [[ -z ${1} || ${1} -lt 60 ]] ;then
        min=0 ; secs="${1}"
    else
        time_mins=$(echo "scale=2; ${1}/60" | bc)
        min=$(echo ${time_mins} | cut -d'.' -f1)
        secs="0.$(echo ${time_mins} | cut -d'.' -f2)"
        secs=$(echo ${secs}*60|bc|awk '{print int($1+0.5)}')
    fi
    echo "Time Elapsed : ${min} minutes and ${secs} seconds."
}

export CUDA_VISIBLE_DEVICES=0 # choose gpu
export HIP_VISIBLE_DEVICES=0 # choose gpu

#set-up bc (calculator)
cp /tf_benchmarks/bin/bc /usr/bin

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        --vendor)
            VENDOR=$VALUE
            ;;
        --iter)
            COUNT=$VALUE
            ;;
        --mode)
            MODE=$VALUE
            ;;
        --batch)
            BATCH=$VALUE
            ;;
        --activation)
            ACTIVATION=$VALUE
            ;;
        --precision)
            PRECISION=$VALUE
            ;;
        --width)
            WIDTH=$VALUE
            ;;
        --height)
            HEIGHT=$VALUE
            ;;
        --channels)
            CHANNELS=$VALUE
            ;;
        --kernel)
            KERNEL=$VALUE
            ;;
        --stride)
            STRIDE=$VALUE
            ;;
        --dilation)
            DILATION=$VALUE
            ;;   
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

if [ -z "$VENDOR" ]
then
      VENDOR=amd
      echo " SET VENDOR=$VENDOR"
fi


if [ -z "$COUNT" ]
then
      COUNT=100
      echo " SET ITERATIONS=$COUNT"
fi

if [ -z "$BATCH" ]
then
      BATCH=128
      echo " SET BATCH_SIZE=$BATCH"
fi

if [ -z "$ACTIVATION" ]
then
      ACTIVATION=relu
      echo " SET ACTIVATION=$ACTIVATION"
fi

if [ -z "$MODE" ]
then
      MODE=benchmark
      echo " SET MODE=$MODE"
fi

if [ -z "$PRECISION" ]
then
      PRECISION=fp32
      echo " SET PRECISION=$PRECISION"
fi

if [ -z "$WIDTH" ]
then
      WIDTH=128
      echo " SET WIDTH=$WIDTH"
fi

if [ -z "$HEIGHT" ]
then
      HEIGHT=128
      echo " SET HEIGHT=$HEIGHT"
fi

if [ -z "$CHANNELS" ]
then
      CHANNELS=3
      echo " SET CHANNELS=$CHANNELS"
fi

if [ -z "$KERNEL" ]
then
      KERNEL=3
      echo " SET KERNEL=$KERNEL"
fi

if [ -z "$STRIDE" ]
then
      STRIDE=1
      echo " SET STRIDE=$STRIDE"
fi

if [ -z "$DILATION" ]
then
      DILATION=1
      echo " SET DILATION=$DILATION"
fi

starttime=$(date +%s)
# run conv2_layer
output=$(python3 /tf_benchmarks/Conv2Layer/layer_conv2.py --iter=$COUNT --precision=$PRECISION --mode=$MODE --batch_size=$BATCH --activation=$ACTIVATION \
--width=$WIDTH --height=$HEIGHT --channels=$CHANNELS --kernel=$KERNEL --stride=$STRIDE --dilation=$DILATION  &> /tf_benchmarks/Conv2Layer/log.txt)
endtime=$(date +%s)
echo "[Conv2Layer]" >> eval_results.txt
echo "VENDOR=$VENDOR MODE=$MODE PRECISION=$PRECISION ITER=$COUNT BATCH_SIZE=$BATCH ACTIVATION=$ACTIVATION" >> eval_results.txt
echo "WIDTH=$WIDTH HEIGHT=$HEIGHT CHANNELS=$CHANNELS KERNEL=$KERNEL STRIDE=$STRIDE DILATION=$DILATION" >> eval_results.txt
secs_to_human "$(($(date +%s) - ${starttime}))" >> eval_results.txt

