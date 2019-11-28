"""
Simple example tho, this makes a about 7.0MB of single TF log file
"""
import tensorflow as tf
import numpy as np

from tensorboard_tools.utils.eager_util import eager_setup
from tensorboard_tools.utils.abs_path import ROOT_DIR

eager_setup()


if __name__ == '__main__':
    summary_writer = tf.compat.v2.summary.create_file_writer(logdir=ROOT_DIR + "/../logs/simple")
    with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
            for t in range(100000):
                print("Step: {}".format(t))
                x = np.random.randn()
                tf.compat.v2.summary.scalar(name="x/random", data=x, step=t)
