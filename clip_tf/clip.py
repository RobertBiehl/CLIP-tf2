import hashlib
import urllib
import warnings

import tqdm
import os
import tensorflow as tf

from clip_tf.converter import get_cache_path, convert

default_cache_path = os.path.expanduser("~/.cache/clip_tf")


def _load(model_name: str, cache_path: str = None, type: str = None, verify: bool = False) -> tf.keras.Model:
    if cache_path is None:
        cache_path = default_cache_path

    output_path = f"{cache_path}{os.path.sep}CLIP_{model_name}"
    model_path = get_cache_path(model_name, output_path, type)

    # TODO: add a way to create a model object and load weights into it.

    model = None
    if os.path.isdir(model_path):
        model = tf.keras.models.load_model(model_path)

    if model is None:
        convert(model_name, output_path, all=True, should_verify=verify)
        model = tf.keras.models.load_model(model_path)

    return model


def get_model(model_name: str, cache_path: str = None, verify: bool = False) -> tf.keras.Model:
    return _load(model_name, cache_path, verify=verify)


def get_image_encoder(model_name: str, cache_path: str = None, verify: bool = False) -> tf.keras.Model:
    return _load(model_name, cache_path, "image", verify=verify)


def get_text_encoder(model_name: str, cache_path: str = None, verify: bool = False) -> tf.keras.Model:
    return _load(model_name, cache_path, "text", verify=verify)

