"""The layer normalization model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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
flags.DEFINE_integer("batch_size",   64, "Hidden Size")
flags.DEFINE_integer("hidden_size", 1024, "Hidden Size")
flags.DEFINE_string("precision","fp32","precision fp32 or fp16")
flags.DEFINE_string("mode","benchmark","Mode")

def init_weights(shape,precision):
  if precision == "fp32":
      xtype = tf.float32
  else: 
      xtype= tf.float16
  return tf.Variable(tf.random.normal(shape, stddev=0.01, dtype=xtype))


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  # return tf.contrib.layers.layer_norm(
  #     inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
  return tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,  center=False, scale=False)(inputs=input_tensor)

class Lnorm(tf.test.TestCase):
 
  def test_lnorm(self):
   
    batch_size  = FLAGS.batch_size
    hidden_size = FLAGS.hidden_size

    #initialize x_trf 
    x_trf  = init_weights([batch_size,hidden_size],FLAGS.precision)
        
    context_layer_gpu  = layer_norm(x_trf)   
    context_layer_gpu_gradient = tf.gradients(ys=context_layer_gpu,xs=x_trf)
   
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                          tf.compat.v1.local_variables_initializer())
    i = 0   
    with tf.compat.v1.Session() as sess:
      sess.run(init_op)
      with tf.device('/GPU:0'): 
          while i < (FLAGS.iter):
            sess.run(context_layer_gpu_gradient)
            i = i+1
        
    print("Final iteration is ", i)
if __name__ == "__main__":
  tf.test.main()
        

    
