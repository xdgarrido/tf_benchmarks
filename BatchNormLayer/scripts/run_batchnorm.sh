export PATH=$PATH:../bin/
function usage()
{
    echo "Usage:"
    echo ""
    echo "./run_batchnorm.sh"
    echo "\t-h --help"
    echo "\t--iter=$COUNT (number of iterations) "
    echo "\t--size=$SIZE (hidden_size) "
    echo "\t--vendor=$VENDOR (amd or nvidia)"
    echo "\t--mode=$MODE (benchmark or validation)"
    echo "\t--precision=$PRECISION (fp32 or fp16)"
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
#cp /tf_benchmarks/bin/bc /usr/bin

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
        --precision)
            PRECISION=$VALUE
            ;;
        --size)
            SIZE=$VALUE
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
# run batch norm
output=$(python3 /tf_benchmarks/BatchNormLayer/batch_normalization.py --iter=$COUNT  --precision=$PRECISION --mode=$MODE --hidden_size=$SIZE &> /tf_benchmarks/BatchNormLayer/log.txt)
endtime=$(date +%s)
echo "[BATCHNORM]" >> eval_results.txt
echo "VENDOR=$VENDOR" >> eval_results.txt
echo "MODE=$MODE" >> eval_results.txt
echo "ITER=$COUNT" >> eval_results.txt
echo "PRECISION=$PRECISION" >> eval_results.txt
echo "HIDDEN_SIZE=$SIZE" >> eval_results.txt
#secs_to_human "$(($(date +%s) - ${starttime}))" >> eval_results.txt
echo "ELAPSED_TIME(in secs)=$((${endtime} - ${starttime}))" >> eval_results.txt