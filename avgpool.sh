#!/bin/bash
MODE=1

PWD=`pwd`
DOCKER_DIR=/`basename $PWD`
export CUDA_VISIBLE_DEVICES=0 # choose gpu
export HIP_VISIBLE_DEVICES=0 # choose gpu
function usage()
{
    echo "Usage:"
    echo ""
    echo "./maxpool.sh"
    echo "\t-h --help"
    echo "\t--vendor=$VENDOR (AMD or NVIDIA)"
    echo "\t--precision=$PRECISION (fp32 or fp16)"
    echo "\t--iter=$COUNT (number of iterations)"
    echo ""
}

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
        --precision)
            PRECISION=$VALUE
            ;;
        --iter)
            COUNT=$VALUE
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

# Get the folders
SCRIPTPATH=$(dirname $(realpath $0))
CODE_DIR=$SCRIPTPATH/..
CODE_DIR_INSIDE=/data


if [ "$VENDOR" = "AMD" ]
then    
        CTNRNAME=GpuContainer
        IMAGE="devenamd/tensorflow:rocm37-rocmfork-horovod-200805"
        echo "Starting $CTNRNAME"
        docker stop GpuContainer
        docker run --name $CTNRNAME -it -d --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --user $(id -u):$(id -g) -w $CODE_DIR_INSIDE -v $CODE_DIR:$CODE_DIR_INSIDE $IMAGE
        echo "$VENDOR"
fi

if [ "$VENDOR" = "NVIDIA" ]
then
        CTNRNAME=GpuContainer
        IMAGE="nvcr.io/nvidia/tensorflow:20.06-tf2-py3"
        echo "Starting $CTNRNAME"
        docker stop GpuContainer
        docker run --name $CTNRNAME -it -d --rm --gpus 1 --network=host --shm-size=16g  --ulimit memlock=-1 --ulimit stack=67108864 -w $CODE_DIR_INSIDE -v $CODE_DIR:$CODE_DIR_INSIDE --user $(id -u):$(id -g) --privileged --device=/dev/kfd --device=/dev/dri --group-add video  --security-opt seccomp=unconfined $IMAGE 
        echo "$VENDOR"
fi

let "count = 0"
CHANNELS=3
for BATCH in 32 64 128
  do 
    for WIDTH in  64 96 128 192 256
    do 
      for HEIGHT in 64 96 128 192 256
        do 
           for POOL in 2 4 8
           do   
              for STRIDE in 1 2 
              do    
                echo $BATCH:$WIDTH:$HEIGHT:$POOL:$STRIDE:$PRECISION
                echo "[AVGLAYER]" >> $SCRIPTPATH/eval_results.txt
                echo "VENDOR=$VENDOR" >> $SCRIPTPATH/eval_results.txt
                echo "MODE=$MODE" >> $SCRIPTPATH/eval_results.txt
                echo "PRECISION=$PRECISION" >> $SCRIPTPATH/eval_results.txt
                echo "ITER=$COUNT" >> $SCRIPTPATH/eval_results.txt
                echo "BATCH_SIZE=$BATCH" >> $SCRIPTPATH/eval_results.txt
                echo "WIDTH=$WIDTH" >> $SCRIPTPATH/eval_results.txt
                echo "HEIGHT=$HEIGHT" >> $SCRIPTPATH/eval_results.txt
                echo "CHANNELS=$CHANNELS" >> $SCRIPTPATH/eval_results.txt
                echo "STRIDE=$STRIDE" >> $SCRIPTPATH/eval_results.txt
                echo "POOL=$POOL" >> $SCRIPTPATH/eval_results.txt
                starttime=$(date +%s)
                # run conv2_layer
                docker exec $CTNRNAME python3 $CODE_DIR_INSIDE/tf_benchmarks/AvgpoolLayer/layer_avg.py --iter=$COUNT --precision=$PRECISION --mode=$MODE --batch_size=$BATCH  \
                    --width=$WIDTH --height=$HEIGHT --channels=$CHANNELS --pool=$POOL --stride=$STRIDE  2>&1 | tee $SCRIPTPATH/AvgpoolLayer/log.txt
                endtime=$(date +%s)
                echo "ELAPSED_TIME(in secs)=$((${endtime} - ${starttime}))"  >> $SCRIPTPATH/eval_results.txt
                let "count++"
              done 
           done 
        done 
    done 
done

docker stop GpuContainer
echo "Parameter Space: $count"


