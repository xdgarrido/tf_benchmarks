export PATH=$PATH:../bin/
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
function usage()
{
    echo "Usage:"
    echo ""
    echo "./run_optimizer.sh"
    echo "\t-h --help"
    echo "\t--iter=$COUNT (number of iterations) "
    echo "\t--optimizer=$TYPE (adam, lamb, nadam or nlamb)"
    echo "\t--vendor=$VENDOR (amd or nvidia)"
    echo "\t--netsize=$SIZE (network size)"
    echo "\t--mode=$MODE (benchmark or validation)"
    echo ""
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

if [ -z "$VENDOR" ]
then
      VENDOR=amd
      echo " SET VENDOR=$VENDOR"
fi

if [ -z "$SIZE" ]
then
      SIZE=10
      echo " SET SIZE=$SIZE"
fi

if [ -z "$COUNT" ]
then
      COUNT=100
      echo " SET ITERATIONS=$COUNT"
fi

if [ -z "$TYPE" ]
then
      TYPE=adam
      echo " SET OPTIMIZER=$TYPE"
fi

if [ -z "$MODE" ]
then
      MODE=benchmark
      echo " SET MODE=$MODE"
fi

starttime=$(date +%s)
# run optimization
output=$(python3 /tf_benchmarks/Optimizers/optimization.py --iter=$COUNT --netsize=$SIZE --mode=$MODE --optimizer_type=$TYPE &> /tf_benchmarks/Optimizers/log.txt)
endtime=$(date +%s)
echo "[OPTIMIZERS]" >> eval_results.txt
echo "VENDOR=$VENDOR" >> eval_results.txt
echo "MODE=$MODE" >> eval_results.txt
echo "ITER=$COUNT"  >> eval_results.txt
echo "NETSIZE=$SIZE" >> eval_results.txt
echo "OPTIMIZER=$TYPE" >> eval_results.txt
#secs_to_human "$(($(date +%s) - ${starttime}))" >> eval_results.txt
echo "ELAPSED_TIME(in secs)=$((${endtime} - ${starttime}))" >> eval_results.txt