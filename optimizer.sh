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
    echo "./dense.sh"
    echo "\t-h --help"
     echo "\t--iter=$COUNT (number of iterations) "
    echo "\t--optimizer=$TYPE (adam, lamb, nadam or nlamb)"
    echo "\t--vendor=$VENDOR (amd or nvidia)"
    echo "\t--netsize=$SIZE (network size)"
    echo "\t--mode=$MODE (benchmark or validation)"
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
        --netsize)
            SIZE=$VALUE
            ;;
        --iter)
            COUNT=$VALUE
            ;;
        --optimizer)
            TYPE=$VALUE
            ;;
        --mode)
            MODE=$VALUE
            ;;
        --profile)
            PROFILE=$VALUE
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
    for SIZE in  500000 1000000 5000000 10000000 50000000 100000000 500000000 
     do 
      for TYPE in adam lamb nadam nlamb
        do
                echo $SIZE:$TYPE
                echo "[OPTIMIZERS]" >> $SCRIPTPATH/eval_results.txt
                echo "VENDOR=$VENDOR" >> $SCRIPTPATH/eval_results.txt
                echo "PRECISION=fp32" >> $SCRIPTPATH/eval_results.txt
                echo "MODE=$MODE" >> $SCRIPTPATH/eval_results.txt
                echo "ITER=$COUNT"  >> $SCRIPTPATH/eval_results.txt
                echo "NETSIZE=$SIZE" >> $SCRIPTPATH/eval_results.txt
                echo "OPTIMIZER=$TYPE" >> $SCRIPTPATH/eval_results.txt
                starttime=$(date +%s)
                # run dense layer
                docker exec $CTNRNAME python3 $CODE_DIR_INSIDE/tf_benchmarks/Optimizers/optimization.py --iter=$COUNT --mode=$MODE --iter=$COUNT --netsize=$SIZE --optimizer_type=$TYPE 2>&1 | tee $SCRIPTPATH/Optimizers/log.txt
                endtime=$(date +%s)
                #secs_to_human "$(($(date +%s) - ${starttime}))" >> eval_results.txt
                echo "ELAPSED_TIME(in secs)=$((${endtime} - ${starttime}))" >> $SCRIPTPATH/eval_results.txt
                let "count++"
        done 
    done


docker stop GpuContainer
echo "Parameter Space: $count"


