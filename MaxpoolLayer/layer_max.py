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
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

tf.compat.v1.disable_v2_behavior()


#input parameters
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


#input parameters
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("iter",  5000, "Total number of iterations")
flags.DEFINE_integer("width",  128, "img width")
flags.DEFINE_integer("height", 128, "img height")
flags.DEFINE_integer("channels", 3, "numbel of color channels")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("stride",1, "convolution stride factor")
flags.DEFINE_integer("pool",2,"pool_size")
flags.DEFINE_string("precision","fp32","precision fp32 or fp16")
flags.DEFINE_string("mode","benchmark","Mode")

def init_weights(shape,precision):
  if precision == "fp32":
      xtype = tf.float32
  else: 
      xtype= tf.float16
  return tf.Variable(tf.random.normal(shape, stddev=0.01, dtype=xtype))

def layer_max(x, pool_size=(2, 2), strides=(1,1), name=None):
  """Run max pool on the input tensor."""
  return(tf.keras.layers.MaxPool2D(pool_size, strides=None, padding='valid', data_format='channels_last')(x))
  
class MaxLayer(tf.test.TestCase):
 
  def test_maxlayer(self):
   
    batch_size = FLAGS.batch_size
    width = FLAGS.width
    height = FLAGS.height
    channels = FLAGS.channels
    strides = (FLAGS.stride,FLAGS.stride)
    pool_size = (FLAGS.pool,FLAGS.pool)
    # logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    number_of_samples = FLAGS.iter * batch_size
    #initialize x_trf 
    x_trf  = init_weights([batch_size,width,height,channels],FLAGS.precision)
      
    context_layer_gpu  = layer_max(x_trf, pool_size, strides)
    context_layer_gpu_gradient = tf.gradients(ys=context_layer_gpu,xs=x_trf)
   
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                          tf.compat.v1.local_variables_initializer())
    i = 0   
    with tf.compat.v1.Session(config=config) as sess:
      sess.run(init_op)
      with tf.device('/GPU:0'): 
          #tf.profiler.experimental.start(logs)
          while i < (number_of_samples):
            sess.run(context_layer_gpu_gradient)
            i = i+batch_size
          #tf.profiler.experimental.stop(logs)
        
    print("Final iteration is ", i)
if __name__ == "__main__":
  tf.test.main()
        

    
