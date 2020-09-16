#!/bin/bash
ITER3=5000
VENDOR=AMD
let "count = 0"
for BATCH in  16 32 64 128 256 512 
  do 
    for SIZE in  32 64 128 256 512 1024 2048 4096 
     do 
      for ACTIVATION in linear relu sigmoid hard_sigmoid softmax softplus swish softsign tanh selu elu exponential
        do
          for PRECISION in fp32
            do
                echo $BATCH:$SIZE:$ACTIVATION:$PRECISION
                ./DenseLayer/scripts/run_layerdense.sh --iter=$ITER3 --vendor=$VENDOR --precision=$PRECISION --mode=$MODE --batch=$BATCH --activation=$ACTIVATION --size=$SIZE
                let "count++"
            done
        done 
    done
done 


echo "Parameter Space: $count"


