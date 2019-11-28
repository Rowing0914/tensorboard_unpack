import os
import io
import fnmatch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from array import array
from tensorflow.python.framework import tensor_util
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tensorboard_tools.utils.abs_path import ROOT_DIR

EVENTS_FILE_PREFIX = "events.*"
IMAGE_FILE_EXTENSION = ".png"
UNPACK_DIR_NAME = "_unpacked"
TFBOARD_SCALAR = "scalars"
TFBOARD_IMAGES = "images"


def _make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def _find(pattern, path):
    # https://stackoverflow.com/a/1724723
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def unpack_board(dir_name):
    event_file = _find(EVENTS_FILE_PREFIX, dir_name)
    assert len(event_file) == 1
    bucket_data, bucket_type = dict(), dict()
    for event in tf.compat.v1.train.summary_iterator(os.path.join(dir_name, event_file[0])):
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/summary_iterator
        for value in event.summary.value:
            if value.tag not in bucket_data.keys():
                bucket_data[value.tag] = list()
                bucket_type[value.tag] = "{}".format(value.metadata.plugin_data.plugin_name)

            _value = tensor_util.MakeNdarray(value.tensor)
            bucket_data[value.tag].append(_value)

    for key, value in bucket_data.items():
        file_name = "_".join(key.split("/"))
        if bucket_type[key] == TFBOARD_SCALAR:
            print("=== Save: {} ===".format(key))
            _data = np.asarray(value)
            _dir_name = os.path.join(dir_name, UNPACK_DIR_NAME, TFBOARD_SCALAR)
            _make_dir(dir_name=_dir_name)
            np.save(arr=_data, file=os.path.join(_dir_name, file_name))
        elif bucket_type[key] == TFBOARD_IMAGES:
            print("=== Save: {} ===".format(key))
            _dir_name = os.path.join(dir_name, UNPACK_DIR_NAME, TFBOARD_IMAGES)
            _make_dir(dir_name=_dir_name)
            bytes = bytearray(value[0][2])
            image = Image.open(io.BytesIO(bytes))
            file_path = os.path.join(_dir_name, file_name + IMAGE_FILE_EXTENSION)
            image.save(file_path)
