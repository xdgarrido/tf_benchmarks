"""The optimizer model as implemented by BERT/Trf and related functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
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
flags.DEFINE_string("mode","benchmark","Mode")
flags.DEFINE_integer("netsize",100000000,"Network Size")
flags.DEFINE_string("optimizer_type","adam","Optimizer Type: adam, lamb, nadam, nlamb")

class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.compat.v1.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())
      v = tf.compat.v1.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())

      # Standard Adam update.
      next_m = (
        tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
        tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
        [param.assign(next_param),
          m.assign(next_m),
          v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


class LAMBOptimizer(tf.compat.v1.train.Optimizer):
  """
  LAMBOptimizer optimizer. 
  https://github.com/ymcui/LAMB_Optimizer_TF

  # References
  - Large Batch Optimization for Deep Learning: Training BERT in 76 minutes. https://arxiv.org/abs/1904.00962v3
  - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805

  # Parameters
  - There is nothing special, just the same as `AdamWeightDecayOptimizer`.
  """

  def __init__(self,
              learning_rate,
              weight_decay_rate=0.01,
              beta_1=0.9,
              beta_2=0.999,
              epsilon=1e-6,
              exclude_from_weight_decay=None,
              name="LAMBOptimizer"):
    """Constructs a LAMBOptimizer."""
    super(LAMBOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.compat.v1.get_variable(
          name=param_name + "/lamb_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())
      v = tf.compat.v1.get_variable(
          name=param_name + "/lamb_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      ############## BELOW ARE THE SPECIFIC PARTS FOR LAMB ##############

      # Note: Here are two choices for scaling function \phi(z)
      # minmax:   \phi(z) = min(max(z, \gamma_l), \gamma_u)
      # identity: \phi(z) = z
      # The authors does not mention what is \gamma_l and \gamma_u
      # UPDATE: after asking authors, they provide me the code below.
      # ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
      #      math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

      r1 = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(param)))
      r2 = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(update)))

      r = tf.compat.v1.where(tf.greater(r1, 0.0), tf.compat.v1.where(
        tf.greater(r2, 0.0), r1/r2, 1.0), 1.0)

      eta = self.learning_rate * r

      update_with_lr = eta * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
class NadamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
  """
  Optimizer that implements the Nadam algorithm.  Nadam is Adam with
  Nesterov momentum.
  
  A basic Nadam optimizer that includes "correct" L2 weight decay.

  References
    See [Dozat, T., 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).
    https://github.com/tdozat/Optimization/blob/master/tensorflow/nadam.py
    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam
  """

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.00,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="NadamWeightDecayOptimizer"):
    """Constructs a NadamWeightDecayOptimizer."""
    super(NadamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    # get the local step 
    steps = tf.cast(global_step, tf.float32) + 1.
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)
      
      m = tf.compat.v1.get_variable(
          name=param_name + "/nadam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())
      v = tf.compat.v1.get_variable(
          name=param_name + "/nadam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())

      # Standard Adam update.
      next_m = (tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad)))
     
      # We could use momentum scheduling variable 
      # mu_t   = self.beta_1 * (1. - 0.5 * (0.96**(0.004*steps))) 
      # instead we use constant scheduling so mu_t = self_beta1       
     
      beta1_correction = 1./(1. - (self.beta_1 ** steps))
      beta1_correction_tp1 = 1./(1. - (self.beta_1 ** (steps+1)))
      beta2_correction = 1./(1. - (self.beta_2 ** steps))

      next_m_unbiased = tf.multiply(beta1_correction_tp1,next_m)
      next_v_unbiased = tf.multiply(beta2_correction,next_v)
      # Nesterov addition moment calculation
      
      next_m_nesterov = (tf.multiply(self.beta_1, next_m_unbiased) + tf.multiply((1.0-self.beta_1)*beta1_correction,grad))

      update = next_m_nesterov / (tf.sqrt(next_v_unbiased) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
        [param.assign(next_param),
          m.assign(next_m),
          v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name

class NlambOptimizer(tf.compat.v1.train.Optimizer):
  """
  Optimizer that implements the NLAMB algorithm.  Nlamb is Lamb with
  Nesterov momentum.
  
  A basic Nlamb optimizer that includes "correct" L2 weight decay.

  References
    See [Dozat, T., 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).
     https://github.com/tdozat/Optimization/blob/master/tensorflow/nadam.py
     https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam
  """
  
  def __init__(self,
              learning_rate,
              weight_decay_rate=0.00,
              beta_1=0.9,
              beta_2=0.999,
              epsilon=1e-6,
              exclude_from_weight_decay=None,
              name="NlambOptimizer"):
    """Constructs a NlamOptimizer."""
    super(NlambOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    # get the local step 
    steps = tf.cast(global_step, tf.float32) + 1.
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.compat.v1.get_variable(
          name=param_name + "/nlamb_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())
      v = tf.compat.v1.get_variable(
          name=param_name + "/nlamb_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))
     
      beta1_correction     = 1./(1. - (self.beta_1 ** steps))
      beta1_correction_tp1 = 1./(1. - (self.beta_1 ** (steps+1)))
      beta2_correction     = 1./(1. - (self.beta_2 ** steps))

      next_m_unbiased = tf.multiply(beta1_correction_tp1,next_m)
      next_v_unbiased = tf.multiply(beta2_correction,next_v)
      # Nesterov addition moment calculation
      
      next_m_nesterov = (tf.multiply(self.beta_1, next_m_unbiased) + tf.multiply((1.0-self.beta_1)*beta1_correction,grad))
    
      update = next_m_nesterov / (tf.sqrt(next_v_unbiased) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      ############## BELOW ARE THE SPECIFIC PARTS FOR LAMB ##############

      # Note: Here are two choices for scaling function \phi(z)
      # minmax:   \phi(z) = min(max(z, \gamma_l), \gamma_u)
      # identity: \phi(z) = z
      # The authors does not mention what is \gamma_l and \gamma_u
      # UPDATE: after asking authors, they provide me the code below.
      # ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
      #      math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

      r1 = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(param)))
      r2 = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(update)))

      r = tf.compat.v1.where(tf.greater(r1, 0.0), tf.compat.v1.where(
        tf.greater(r2, 0.0), r1/r2, 1.0), 1.0)

      eta = self.learning_rate * r

      update_with_lr = eta * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name

def init_rand_variable(shape):
    return tf.Variable(tf.random.uniform(shape, minval=-0.7, maxval=1.3))


def init_ones(shape):
    return tf.ones(shape)

class Optimization(tf.test.TestCase):
 
  def test_optimizer(self):
    w = init_rand_variable([FLAGS.netsize])
    target  = init_ones([FLAGS.netsize])
      
    loss  = tf.reduce_mean(input_tensor=tf.square(target - w))
    tvars = tf.compat.v1.trainable_variables()
    grads = tf.gradients(ys=loss, xs=tvars)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    if FLAGS.optimizer_type == "adam":
        print("Initializing ADAM Optimizer")
        optimizer = AdamWeightDecayOptimizer(learning_rate=0.2)
    elif FLAGS.optimizer_type == "lamb":
        print("Initializing LAMB Optimizer")
        optimizer = LAMBOptimizer(learning_rate=0.2)
    elif FLAGS.optimizer_type == "nadam":
        print("Initializing NADAM Optimizer")
        optimizer = NadamWeightDecayOptimizer(learning_rate=0.2)
    else:
        print("Initializing NLAMB Optimizer")
        optimizer = NlambOptimizer(learning_rate=0.2)

    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step)
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                          tf.compat.v1.local_variables_initializer())
        
    with self.session() as sess:
        with tf.device('/GPU:0'): 
            sess.run(init_op)
            for _ in range(FLAGS.iter):
              sess.run(train_op)
            if FLAGS.mode == "validation" :
              w_final = sess.run(w)
              self.assertAllClose(w_final.flat, target, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  tf.test.main()
