#!/bin/bash
ITER3=5000
VENDOR=AMD
PRECISION=fp32
let "count = 0"
for batch in {16..32..16}
  do 
    for width in {16..32..16}
    do 
      for height in {16..32..16}
        do 
           for kernel in {3..11..2}
           do   
              for stride in {1..4..1}
              do 
                  for dilation in {1..4..1}
                  do 
                      for activation in linear relu sigmoid hard_sigmoid softmax softplus swish softsign tanh selu elu exponential
                      do
                          for precision in $PRECISION
                          do
                            echo $batch:$width:$height:$kernel$stride:$dilation:$activation:$precision
                            ./Conv2Layer/scripts/run_conv2layer.sh --iter=$ITER3 --batch=$batch  --width=$width --height=$height --channels=3 --kernel=$kernel --stride=$stride --dilation=$dilation --activation=$activation --precision=$precision --vendor=$VENDOR
                            let "count++"
                          done
                      done 
                  done
              done 
           done 
        done 
    done 
done

echo "Parameter Space: $count"


