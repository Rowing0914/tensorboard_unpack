import tensorflow as tf
from tensorboard_tools.utils.eager_util import eager_setup

eager_setup()
assert tf.executing_eagerly()