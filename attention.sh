#!/bin/bash
COUNT=100
VENDOR=AMD
PRECISION=fp32
MODE=1

if [ "$VENDOR" = "AMD" ]
then
        IMAGE="devenamd/tensorflow:rocm37-rocmfork-horovod-200805"
fi

if [ "$VENDOR" = "NVIDIA" ]
then
        IMAGE="nvcr.io/nvidia/tensorflow:20.06-tf2-py3"
fi


# Get the folders
# Get the folders
SCRIPTPATH=$(dirname $(realpath $0))
CODE_DIR=$SCRIPTPATH/..
CODE_DIR_INSIDE=/data
CTNRNAME=GpuContainer

export HIP_VISIBLE_DEVICES=0 # choose gpu
echo "Starting $CTNRNAME"
docker stop GpuContainer
docker run --name $CTNRNAME -it -d --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host \
--shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --user $(id -u):$(id -g) \
-w $CODE_DIR_INSIDE -v $CODE_DIR:$CODE_DIR_INSIDE $IMAGE
let "count = 0"
for BATCH in  8 #16 32 
  do 
    for LENGTH in 32 #64 128 256 512 
     do 
      for HEADS in 8 #16 32 64
        do
          for SIZE in 16 #32 64 
            do
                echo $BATCH:$LENGTH:$HEADS:$SIZE
                # run pretraining
                echo "[ATTENTIONHEAD]" >> $SCRIPTPATH/eval_results.txt
                echo "VENDOR=$VENDOR" >> $SCRIPTPATH/eval_results.txt
                echo "MODE=$MODE" >> $SCRIPTPATH/eval_results.txt
                echo "ITER=$COUNT" >> $SCRIPTPATH/eval_results.txt
                echo "PRECISION=$PRECISION" >> $SCRIPTPATH/eval_results.txt
                echo "BATCH_SIZE=$BATCH" >> $SCRIPTPATH/eval_results.txt
                echo "SEQ_LENGTH=$LENGTH" >> $SCRIPTPATH/eval_results.txt
                echo "HEADS=$HEADS" >> $SCRIPTPATH/eval_results.txt
                echo "SIZE_ATTENTION_HEAD=$SIZE" >> $SCRIPTPATH/eval_results.txt
                starttime=$(date +%s)
                docker exec $CTNRNAME python3 $CODE_DIR_INSIDE/tf_benchmarks/Attention/attention.py --iter=$COUNT \
                --seq_length=$LENGTH --batch=$BATCH --precision=$PRECISION --num_attention_heads=$HEADS --attention_head_size=$SIZE \
                --mode=$MODE 2>&1 | tee $SCRIPTPATH/Attention/log.txt
                endtime=$(date +%s)
                echo "ELAPSED_TIME(in secs)=$((${endtime} - ${starttime}))" >> $SCRIPTPATH/eval_results.txt
                let "count++"
            done
        done 
    done
done 
docker stop GpuContainer
echo "Parameter Space: $count"


