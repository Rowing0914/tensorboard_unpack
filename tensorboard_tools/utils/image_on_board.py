import io
import tensorflow as tf
import matplotlib.pyplot as plt


def _preprocess_image(figure):
    """put the image on memory as PNG and convert it into TF image format"""
    # save the plot as PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # close the figure
    plt.close(figure)
    buf.seek(0)
    # convert PNG to TF image format
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def display_image_on_board(figure, name="plot", ts=None):
    """plot the image on TF board"""
    image = _preprocess_image(figure=figure)
    tag, name = name.split("/")[0], "/".join(name.split("/")[1:])
    with tf.name_scope(tag):
        if ts is not None:
            tf.compat.v2.summary.image(name, image, step=ts)
        else:
            tf.compat.v2.summary.image(name, image, step=tf.compat.v1.train.get_global_step())


def display_images_on_board(figures, names, ts=None):
    """plot the collection of images on TF board"""
    assert len(figures) == len(names)
    for figure, name in zip(figures, names):
        image = _preprocess_image(figure=figure)
        if ts:
            tf.compat.v2.summary.image(name, image, step=ts)
        else:
            tf.compat.v2.summary.image(name, image, step=tf.compat.v1.train.get_global_step())
