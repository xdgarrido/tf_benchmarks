export PATH=$PATH:../bin/
function usage()
{
    echo "Usage:"
    echo ""
    echo "./run_layerdense.sh"
    echo "\t-h --help"
    echo "\t--iter=$COUNT (number of iterations) "
    echo "\t--size=$SIZE (hidden_size) "
    echo "\t--batch=$BATCH (batch_size) "
    echo "\t--activation=$ACTIVATION (relu,sigmoid,softmax,softplus,softsign,tanh,selu,elu,exponential) "
    echo "\t--vendor=$VENDOR (amd or nvidia)"
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
        --size)
            SIZE=$VALUE
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

if [ -z "$SIZE" ]
then
      SIZE=1024
      echo " SET HIDDEN_SIZE=$SIZE"
fi

if [ -z "$BATCH" ]
then
      BATCH=512
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

starttime=$(date +%s)
# run dense layer
output=$(python3 /tf_benchmarks/DenseLayer/layer_dense.py --iter=$COUNT --precision=$PRECISION --mode=$MODE --batch_size=$BATCH --activation=$ACTIVATION --hidden_size=$SIZE &> /tf_benchmarks/DenseLayer/log.txt)
endtime=$(date +%s)
echo "[DenseLayer]" >> /tf_benchmarks/DenseLayer/eval_results.txt
echo "VENDOR=$VENDOR MODE=$MODE PRECISION=$PRECISION ITER=$COUNT HIDDEN_SIZE=$SIZE BATCH_SIZE=$BATCH ACTIVATION=$ACTIVATION" >>  /tf_benchmarks/DenseLayer/eval_results.txt
secs_to_human "$(($(date +%s) - ${starttime}))" >> /tf_benchmarks/DenseLayer/eval_results.txt

