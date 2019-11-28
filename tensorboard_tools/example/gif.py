import gym
import numpy as np
import tensorflow as tf

from tensorboard_tools.utils.eager_util import eager_setup
from tensorboard_tools.utils.abs_path import ROOT_DIR
from tensorboard_tools.utils.gif_on_board import gif_on_board


eager_setup()

EPISODE_LENGTH = 100


def roll_out(env_name="PongNoFrameskip-v4"):
    env = gym.make(env_name)
    env.reset()
    buffer = list()
    for _ in range(EPISODE_LENGTH):
        image = env.render(mode="rgb_array")
        buffer.append(image)
        _, _, done, _ = env.step(env.action_space.sample())
        if done: break
    env.close()
    return buffer


def test_gif(buffer):
    # You can watch the animation with matplotlib
    import matplotlib.pyplot as plt
    for i in range(len(buffer)):
        plt.imshow(buffer[i])
        plt.draw()
        plt.pause(0.000001)
        plt.clf()


if __name__ == '__main__':
    buffer = roll_out(env_name="PongNoFrameskip-v4")
    # test_gif(buffer=buffer)  # this takes time to finish...

    summary_writer = tf.compat.v2.summary.create_file_writer(logdir=ROOT_DIR + "/../logs/gif")
    with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
            gif_on_board(images=np.asarray(buffer), name="Test/gif")
