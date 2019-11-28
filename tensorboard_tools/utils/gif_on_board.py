import tensorflow as tf

from tensorboard_tools.utils.gif_summary import gif_summary_v2

FPS = 25


def _gif_summary(name, images, fps, saturate=False, step=None):
    """ Create the summary of Gif Animation of images on Tensorboard """
    # Recover RGB values: [0-1] => [1-255]
    images = tf.image.convert_image_dtype(images, tf.uint8, saturate=saturate)
    output = tf.concat(tf.unstack(images), axis=2)[None]
    gif_summary_v2(name, output, 1, fps, step=step)


def gif_on_board(images, name, ts=None):
    """
    This visualises the reconstructed images
    :param: images: input images (len_seq, w, h, c)
    """
    tag, name = name.split("/")[0], "/".join(name.split("/")[1:])
    with tf.name_scope(tag):
        if ts is not None:
            _gif_summary(name, images[None, ...], FPS, saturate=True, step=ts)
        else:
            _gif_summary(name, images[None, ...], FPS, saturate=True, step=tf.compat.v1.train.get_global_step())
