"""The layer dense model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import numpy as np
import os
# enable just error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

tf.compat.v1.disable_v2_behavior()


#input parameters
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("iter", 5000, "Total number of iterations")
flags.DEFINE_integer("hidden_size", 1024, "Hidden Size")
flags.DEFINE_integer("batch_size", 128, "Hidden Size")
flags.DEFINE_string("precision","fp32","precision fp32 or fp16")
flags.DEFINE_string("activation","relu","activation type")
flags.DEFINE_string("mode","benchmark","Mode")

def init_weights(shape,precision):
  if precision == "fp32":
      xtype = tf.float32
  else: 
      xtype= tf.float16
  return tf.Variable(tf.random.normal(shape, stddev=0.01, dtype=xtype))

def layer_dense(x, hidden_size, activation='relu', name=None):
  """Run dense layer on the input tensor."""
  
  return (tf.keras.layers.Dense(hidden_size, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x))
  

class DenseLayer(tf.test.TestCase):
 
  def test_denselayer(self):
   
    hidden_size = FLAGS.hidden_size
    batch_size = FLAGS.batch_size
    activation = FLAGS.activation
    # number_of_samples = hidden_size * batch_size
    #initialize x_trf 
    x_trf  = init_weights([batch_size,hidden_size],FLAGS.precision)
    #logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")    
    context_layer_gpu  = layer_dense(x_trf, hidden_size, activation)   
    context_layer_gpu_gradient = tf.gradients(ys=context_layer_gpu,xs=x_trf)
   
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                          tf.compat.v1.local_variables_initializer())
    i = 0   
    with tf.compat.v1.Session() as sess:
      sess.run(init_op)
      with tf.device('/GPU:0'): 
          #tf.profiler.experimental.start(logs)
          while i < (FLAGS.iter):
            sess.run(context_layer_gpu_gradient)
            i = i+1
          #tf.profiler.experimental.stop(logs) 
    print("Final iteration is ", i)
if __name__ == "__main__":
  tf.test.main()
        

    
