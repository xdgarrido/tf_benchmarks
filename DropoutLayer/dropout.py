
"""The dropout model."""
import sys
import random
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
flags.DEFINE_integer("batch", 6, "Batch size")
flags.DEFINE_integer("seq_length", 512, "Sequence Length")
flags.DEFINE_integer("attention_heads",16,"Number of attention heads")
flags.DEFINE_string("precision","fp32","precision fp32 or fp16")
flags.DEFINE_string("mode","benchmark","Mode")

def init_ones(shape,precision):
    if precision == "fp32":
      xtype = tf.float32
    else: 
      xtype= tf.float16
    return tf.ones(shape,dtype=xtype)


def dropout(input_tensor, dropout_prob, seed):
    """Perform dropout.
    """
    return tf.keras.layers.Dropout(rate=dropout_prob, seed=seed)(input_tensor)

class Drop(tf.test.TestCase):
 
  def test_drop(self):

    # batch and seq size that fit into a single GPU collected from https://github.com/ROCmSoftwarePlatform/BERT#out-of-memory-issues
    batch_size = FLAGS.batch
    seq_length = FLAGS.seq_length

    # number of heads for BERT base model collected from https://github.com/ROCmSoftwarePlatform/BERT#pre-trained-models
    num_attention_heads = FLAGS.attention_heads

    # default dropout prob in BERT model collected from https://github.com/ROCmSoftwarePlatform/BERT/blob/bee6030e31e42a9394ac567da170a89a98d2062f/modeling.py#L42
    attention_probs_dropout_prob = 0.1

    # initialize atttention_scores
    attention_probs = init_ones([batch_size, num_attention_heads, seq_length, seq_length],FLAGS.precision)

    seed = random.randint(0, sys.maxsize)
        
    attention_probs_dropout_gpu = dropout(
            attention_probs, attention_probs_dropout_prob, seed=seed)

    attention_probs_dropout_gpu_gradient = tf.gradients(
            ys=attention_probs_dropout_gpu, xs=attention_probs)
   
    init_op = tf.group(tf.compat.v1.global_variables_initializer())
    i = 0   
    with tf.compat.v1.Session() as sess:
      sess.run(init_op)
      with tf.device('/GPU:0'): 
          while i < (FLAGS.iter):
            sess.run(attention_probs_dropout_gpu_gradient)
            i = i+1
        
    print("Final iteration is ", i)

if __name__ == "__main__":
  tf.test.main()


