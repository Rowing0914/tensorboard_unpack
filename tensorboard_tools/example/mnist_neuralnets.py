"""
Training NN on MNIST dataset
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tensorboard_tools.utils.eager_util import eager_setup
from tensorboard_tools.utils.abs_path import ROOT_DIR
from tensorboard_tools.utils.image_on_board import display_image_on_board


eager_setup()


def get_dataset():
    """get a dataset using mnist API"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # normalise the images
    return (x_train, y_train), (x_test, y_test)


class Network(tf.Module):
    """simple neural network without CNN or other heavy stuff"""

    def __init__(self):
        super(Network, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.output = tf.keras.layers.Dense(10, activation='softmax')

    def __call__(self, inputs):
        out = self.flatten(inputs)
        out = self.dense(out)
        out = self.dropout(out)
        out = self.output(out)
        return out


class Model(object):
    def __init__(self, network, metrics, epoch=10):
        self.epoch = epoch
        self.network = network
        self.metrics = metrics
        self.optimiser = tf.keras.optimizers.Adam()
        self.global_ts = tf.compat.v1.train.create_global_step()

    def train(self, x, y_true):
        """main training function"""
        losses, accs = list(), list()
        for epoch in range(self.epoch):
            outputs = self._train(x=x, y_true=y_true)
            print("Gradient-Step: {}, Loss: {}".format(epoch, outputs["loss"].numpy()))
            losses.append(outputs["loss"].numpy())
            accs.append(outputs["acc"].numpy())
            self.global_ts.assign_add(1)
        outputs = {
            "losses": np.asarray(losses),
            "accs": np.asarray(accs)
        }
        return outputs

    @tf.function
    def _train(self, x, y_true):
        """inner training function"""
        with tf.GradientTape() as tape:
            y_pred = self.network(x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred)
            self.metrics.update_state(y_true=y_true, y_pred=tf.argmax(y_pred, axis=-1))

        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimiser.apply_gradients(zip(grads, self.network.trainable_variables))

        with tf.name_scope("train"):
            tf.compat.v2.summary.scalar(name="loss",
                                        data=tf.reduce_mean(loss),
                                        step=tf.compat.v1.train.get_global_step())
            tf.compat.v2.summary.scalar(name="accuracy",
                                        data=self.metrics.result(),
                                        step=tf.compat.v1.train.get_global_step())

        outputs = {
            "loss": tf.reduce_mean(loss),
            "acc": self.metrics.result()
        }
        return outputs

    @tf.function
    def validation(self, x_test, y_test, metric):
        """validating the training"""
        y_pred = self.network(x_test)
        y_pred = tf.argmax(y_pred, axis=-1)
        metric.update_state(y_true=y_test, y_pred=y_pred)
        return metric.result()


def display_image(image, figsize=(10, 10)):
    """temp function for displaying a MNIST image"""
    figure = plt.figure(figsize=figsize)
    image = image.reshape((28, 28))
    plt.imshow(image, cmap='gray')
    return figure


def display_line_graph(data, label, figsize=(10, 10)):
    """temp function for plotting a line graph"""
    figure = plt.figure(figsize=figsize)
    plt.plot(np.arange(data.shape[0]), data, "g")
    plt.legend(handles=[mpatches.Patch(color="g", label=label)])
    plt.grid()
    return figure


if __name__ == '__main__':
    summary_writer = tf.compat.v2.summary.create_file_writer(logdir=ROOT_DIR + "/../logs/mnist_neuralnets")
    train_metric = tf.keras.metrics.Accuracy()
    eval_metric = tf.keras.metrics.Accuracy()
    model = Model(network=Network(),
                  metrics=train_metric,
                  epoch=10)
    (x_train, y_train), (x_test, y_test) = get_dataset()

    with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
            # training and evaluation of the model
            outputs = model.train(x=x_train, y_true=y_train)
            result = model.validation(x_test=x_test,
                                      y_test=y_test,
                                      metric=eval_metric)
            print("Final result: ", result.numpy())

            # create a canvas for sample image in MNIST and put it on TF board
            figure = display_image(image=x_train[0])
            display_image_on_board(figure=figure, name="train/MNIST", ts=tf.compat.v1.train.get_global_step())

            # create a canvas for line graphs and put it on TF board
            losses_fig = display_line_graph(data=outputs["losses"], label="loss")
            display_image_on_board(figure=losses_fig, name="train/loss curve")
            accs_fig = display_line_graph(data=outputs["accs"], label="acc")
            display_image_on_board(figure=accs_fig, name="train/accuracy curve")

            # TODO: this doesn't work at the moment because when we create plot, we overwrite the previous canvas.
            # losses_fig = display_line_graph(data=outputs["losses"], label="loss")
            # accs_fig = display_line_graph(data=outputs["accs"], label="acc")
            # display_images_on_board(figures=[losses_fig, accs_fig], names=["train/loss curve", "train/accuracy curve"])
