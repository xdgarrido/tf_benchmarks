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
flags.DEFINE_integer("kernel", 3, "kernel size")
flags.DEFINE_integer("stride",1, "convolution stride factor")
flags.DEFINE_integer("dilation",1, "convolution dilation factor")
flags.DEFINE_string("precision","fp32","precision fp32 or fp16")
flags.DEFINE_string("activation","relu","activation type")
flags.DEFINE_string("mode","benchmark","Mode")

def init_weights(shape,precision):
  if precision == "fp32":
      xtype = tf.float32
  else: 
      xtype= tf.float16
  return tf.Variable(tf.random.normal(shape, stddev=0.01, dtype=xtype))

def layer_conv2(x, shape, batch_size, kernel_size=(3,3), strides=(1,1), dilation_rate=(1,1), activation='relu', name=None):
  """Run convolution on the input tensor."""
  
  return(tf.keras.layers.Conv2D(batch_size, kernel_size, strides, padding='valid', dilation_rate=(1,1), activation='relu', input_shape=shape[1:])(x))
  
  

class Conv2Layer(tf.test.TestCase):
 
  def test_conv2layer(self):
   
    batch_size = FLAGS.batch_size
    width = FLAGS.width
    height = FLAGS.height
    channels = FLAGS.channels
    activation = FLAGS.activation
    input_shape = (height,width,channels)
    kernel_size = (FLAGS.kernel,FLAGS.kernel)
    strides = (FLAGS.stride,FLAGS.stride)
    dilation_rate = (FLAGS.dilation,FLAGS.dilation)
    # In case of profiling uncomment
    #logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    number_of_samples = FLAGS.iter * batch_size
    #initialize x_trf 
    x_trf  = init_weights([batch_size,width,height,channels],FLAGS.precision)
    
        
    context_layer_gpu  = layer_conv2(x_trf, input_shape, batch_size, kernel_size, strides, dilation_rate, activation)
    context_layer_gpu_gradient = tf.gradients(ys=context_layer_gpu,xs=x_trf)
   
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                          tf.compat.v1.local_variables_initializer())
    i = 0   
    with tf.compat.v1.Session(config=config) as sess:
      sess.run(init_op)
      with tf.device('/GPU:0'): 
           # In case of profiling uncomment
          #tf.profiler.experimental.start(logs)
          while i < (number_of_samples):
            sess.run(context_layer_gpu_gradient)
            i = i+batch_size
          # In case of profiling uncomment
          #tf.profiler.experimental.stop(logs)
        
    print("Final iteration is ", i)
if __name__ == "__main__":
  tf.test.main()
        

    
