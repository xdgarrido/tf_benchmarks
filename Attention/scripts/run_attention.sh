export PATH=$PATH:../bin/
function usage()
{
    echo "Usage:"
    echo ""
    echo "./run_attention.sh"
    echo "\t-h --help"
    echo "\t--iter=$COUNT (number of iterations) "
    echo "\t--vendor=$VENDOR (amd or nvidia)"
    echo "\t--mode=$MODE (benchmark or validation)"
    echo "\t--length=$LENGTH (sequence length) "
    echo "\t--size=$SIZE (size of attention head)"
    echo "\t--batch=$BATCH (batch size)"
    echo "\t--precision=$PRECISION (fp32 or fp16)"
    echo "\t--heads=$HEADS (number of attention heads)"
    echo ""
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
        --size)
            SIZE=$VALUE
            ;;
        --batch)
            BATCH=$VALUE
            ;;
        --precision)
            PRECISION=$VALUE
            ;;
        --length)
            LENGTH=$VALUE
            ;;
        --heads)
            HEADS=$VALUE
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
      COUNT=10
      echo " SET ITERATIONS=$COUNT"
fi


if [ -z "$MODE" ]
then
      MODE=benchmark
      echo " SET MODE=$MODE"
fi

if [ -z "$BATCH" ]
then
      BATCH=8
      echo " SET BATCH=$BATCH"
fi

if [ -z "$PRECISION" ]
then
      PRECISION=fp32
      echo " SET PRECISION=$PRECISION"
fi

if [ -z "$HEADS" ]
then
      HEADS=16
      echo " SET NUM_HEADS=$HEADS"
fi

if [ -z "$SIZE" ]
then
      SIZE=64
      echo " SET SIZE_ATTENTION_HEAD=$SIZE"
fi

if [ -z "$LENGTH" ]
then
      LENGTH=128
      echo " SET SEQ_LENGTH=$LENGTH"
fi

starttime=$(date +%s)
# run attention
output=$(python3 /tf_benchmarks/Attention/attention.py --iter=$COUNT --seq_length=$LENGTH --batch=$BATCH --precision=$PRECISION --num_attention_heads=$HEADS --attention_head_size=$SIZE --mode=$MODE 2>&1 | tee /tf_benchmarks/Attention/log.txt)
endtime=$(date +%s)
echo "[ATTENTIONHEAD]" >> eval_results.txt
echo "VENDOR=$VENDOR MODE=$MODE ITER=$COUNT PRECISION=$PRECISION BATCH_SIZE=$BATCH SEQ_LENGTH=$LENGTH NUM_ATTENTION_HEADS=$HEADS SIZE_ATTENTION_HEAD=$SIZE" >> eval_results.txt
secs_to_human "$(($(date +%s) - ${starttime}))" >> eval_results.txt

