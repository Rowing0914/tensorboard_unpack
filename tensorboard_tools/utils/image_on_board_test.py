import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tensorboard_tools.utils.eager_util import eager_setup
from tensorboard_tools.utils.abs_path import ROOT_DIR
from tensorboard_tools.utils.image_on_board import display_image_on_board


eager_setup()


def display_line_graph(data, label, figsize=(10, 10)):
    """temp function for plotting a line graph"""
    figure = plt.figure(figsize=figsize)
    plt.plot(np.arange(data.shape[0]), data, "g")
    plt.legend(handles=[mpatches.Patch(color="g", label=label)])
    plt.grid()
    return figure


if __name__ == '__main__':
    summary_writer = tf.compat.v2.summary.create_file_writer(logdir=ROOT_DIR + "/../logs/Test/line_graph")

    with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
            x = np.random.randn(100)
            # create a canvas for line graphs and put it on TF board
            figure = display_line_graph(data=x, label="test")
            display_image_on_board(figure=figure, name="test", ts=0)
