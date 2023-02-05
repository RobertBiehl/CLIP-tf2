import hashlib
import os
import re
import sys
import urllib
import warnings
from typing import List
import logging

import numpy as np
import requests
import torch.hub

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

import clip
from PIL import Image

from clip_tf.model import build_model

LOGGER = logging.Logger(__name__)

MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
}

# model input for verification
image_url = "https://github.com/openai/CLIP/blob/main/CLIP.png?raw=true"
text_options = ["a diagram", "a dog", "a cat", "a neural network"]


def download_statedict(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    is_downloaded = False
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            is_downloaded = True
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    if not is_downloaded:
        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    state_dict = torch.jit.load(
        download_target,
        map_location="cuda" if torch.cuda.is_available() else "cpu"
    ).state_dict()
    return state_dict


def load_pytorch_weights(model: keras.Model, state_dict: dict, verbose=False):
    tf_dict = {v.name.replace('.', '/'): v for v in model.weights}

    # (from, to) replacement pairs to convert tensor names from tf to pytorch
    def prepare_key(key):
        repl = [
            ('clip/', ''),
            (':0', ''),
            ('token_embedding', 'token_embedding/weight'),
            # batch norm
            ('gamma', 'weight'), ('beta', 'bias'),
            ('moving_mean', 'running_mean'), ('moving_variance', 'running_var'),
            # conv
            ('kernel', 'weight'),
            # attention (resnet)
            ('mha/key', 'k_proj'), ('mha/query', 'q_proj'), ('mha/value', 'v_proj'), ('mha/attention_output', 'c_proj'),
            # attention (transformer)
            ('attn/key', 'attn/k_proj'), ('attn/query', 'attn/q_proj'), ('attn/value', 'attn/v_proj'),
            ('attn/attention_output', 'attn/out_proj'),
            ('/', '.'),
        ]
        for rep in repl:
            key = key.replace(*rep)
        return key

    # convert existing keys in state_dict, e.g. splitting tensor into multiple
    initial_converters = {
        "in_proj_weight": lambda key, source: dict(
            zip(
                [
                    key.replace('in_proj_weight', 'q_proj.weight'),
                    key.replace('in_proj_weight', 'k_proj.weight'),
                    key.replace('in_proj_weight', 'v_proj.weight')
                ],
                torch.split(source, source.shape[0] // 3, dim=0)
            )
        ),
        "in_proj_bias": lambda key, source: dict(
            zip(
                [
                    key.replace('in_proj_bias', 'q_proj.bias'),
                    key.replace('in_proj_bias', 'k_proj.bias'),
                    key.replace('in_proj_bias', 'v_proj.bias')
                ],
                torch.split(source, source.shape[0] // 3, dim=0)
            )
        ),
    }

    def apply_initial_converters():
        state_dict_keys = list(state_dict.keys())
        for k, fn in initial_converters.items():
            r = re.compile(k)
            matched_keys = filter(r.search, state_dict_keys)
            for matched_key in matched_keys:
                res = fn(matched_key, state_dict[matched_key])
                for key, val in res.items():
                    state_dict[key] = val
                del state_dict[matched_key]

    apply_initial_converters()

    # convert keys when their destination is known, maps to tensor
    def multi_head_attention_weight_conversion(source, dest):
        res = source.T.reshape(tuple(np.array(dest.shape)))
        return res

    contextual_converters = {
        '_proj.weight': multi_head_attention_weight_conversion,
        '_proj.bias': multi_head_attention_weight_conversion,
    }

    def apply_contextual_converters(state_dict_key, source_weights, dest):
        for k, fn in contextual_converters.items():
            r = re.compile(k)
            if r.search(state_dict_key):
                return fn(source_weights, dest), k

        return source_weights, ''

    mapped_keys = set()

    for tf_key in tqdm(tf_dict.keys(), desc="Copying weights"):
        state_dict_key = prepare_key(tf_key)

        dest = tf_dict[tf_key]

        if state_dict_key not in state_dict:
            candidates = [key for key, src in state_dict.items()
                          if key not in mapped_keys and tf.reduce_prod(dest.shape) == np.prod(src.shape)]
            raise ValueError(
                f"'{tf_key}': Missing var {state_dict_key} in state_dict. shape={dest.shape}. candidates={candidates}")

        if state_dict_key in mapped_keys:
            raise ValueError(
                f"'{tf_key}': Duplicate var {state_dict_key} has already been assigned.")
        mapped_keys.add(state_dict_key)

        source_weights = state_dict[state_dict_key].cpu().detach()

        source_weights, converter_name = apply_contextual_converters(state_dict_key, source_weights, dest)

        compatible_weights_default = tf.reduce_all(dest.shape == source_weights.shape)
        compatible_weights_transposed = False
        if len(source_weights.shape) == 4:
            compatible_weights_transposed = tf.reduce_all(dest.shape == source_weights.permute(2, 3, 1, 0).shape)
            if compatible_weights_default and compatible_weights_transposed and len(source_weights.shape) > 1:
                print(
                    f"'{state_dict_key}' -> '{tf_key}': unclear whether shape {source_weights.shape} should be transposed to {dest.shape}",
                    file=sys.stderr)
            if compatible_weights_transposed and not compatible_weights_default:
                source_weights = source_weights.permute(2, 3, 1, 0)
                pass
        if len(source_weights.shape) == 2:
            compatible_weights_transposed = tf.reduce_all(dest.shape == source_weights.permute(1, 0).shape)
            if compatible_weights_default and compatible_weights_transposed and len(source_weights.shape) > 1:
                print(
                    f"'{state_dict_key}' -> '{tf_key}': unclear whether shape {source_weights.shape} should be transposed to {dest.shape}",
                    file=sys.stderr)
            if compatible_weights_transposed and not compatible_weights_default:
                source_weights = source_weights.permute(1, 0)
                pass

        if not (compatible_weights_default or compatible_weights_transposed):
            print(
                f"'{state_dict_key}' -> '{tf_key}': source shape {source_weights.shape} has to be equal to dest shape {dest.shape}",
                file=sys.stderr)

        # assert compatible_weights, f"'{key}': source shape {source_weights.shape} has to be equal to dest shape {dest.shape}"
        if verbose:
            print(
                f"convert '{state_dict_key}' -> '{tf_key}' {source_weights.shape} converter='{converter_name}' {'transposed' if compatible_weights_transposed and not compatible_weights_default else ''}")
        assert isinstance(dest, object)
        dest.assign(source_weights.numpy().astype(np.float32))

    # unmapped_keys = set(state_dict.keys()).difference(mapped_keys)
    # if len(unmapped_keys) > 0:
    #     print("Unmapped keys in state_dict:")
    #     for k in unmapped_keys:
    #         print(f"missing '{k}' -> '?'")
    #
    #     exit(0)


def verify(model_name: str, keras_model: keras.Model, image_url: str, text_options: List[str], verbose: bool = False):
    # load pytorch clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device, jit=False)
    image = preprocess(
        Image.open(requests.get(image_url, stream=True).raw)
    ).unsqueeze(0)
    text = clip.tokenize(text_options)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(
            image.to(device),
            text.to(device)
        )
        torch_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # tf2
    image = image.permute(0, 2, 3, 1).detach().numpy()
    text = text.unsqueeze(0)  # grml... keras doesnt like different cardinality in batch dim
    text = text.detach().numpy().astype(np.int32)
    logits_per_image, logits_per_text = keras_model.predict((image, text))
    tf_probs = tf.nn.softmax(logits_per_image, axis=1)
    tf_probs = np.array(tf_probs)

    if verbose:
        print(f"Classify image: {image_url}")
        print(f"Text options: {text_options}")
        print(f"Pytorch: {torch_probs}")
        print(f"Tensorflow: {tf_probs}")

    assert np.abs(
        torch_probs - tf_probs).sum() < 1e-3, f"PyTorch and Tensorflow results should be almost equal: torch_probs={torch_probs}, tf_probs={tf_probs}"


def get_cache_path(model: str, cache_path: str, type: str = None) -> str:
    sanitized_model_name = model.replace("/", "_")
    if type is not None:
        sanitized_model_name = f"{type}_{sanitized_model_name}"
    return cache_path.format(model=sanitized_model_name)


def convert(model_name: str, output: str, image_output: str = None, text_output: str = None, all: bool = False,
            should_verify: bool = True):
    model_url = MODELS[model_name]
    state_dict = download_statedict(model_url)
    model = build_model(state_dict)

    # predict to build shapes (model.build doesnt work, as it only supports float inputs)
    model.predict((
        np.ones((1, model.image_resolution, model.image_resolution, 3), np.float32),
        np.ones((1, 4, 77), np.int32)
    ))
    load_pytorch_weights(model, state_dict, verbose=False)

    if should_verify:
        LOGGER.info("Verifying converted model...")
        verify(model_name, model, image_url, text_options, verbose=True)

    # create SavedModel
    output_filename = get_cache_path(model_name, output)
    LOGGER.info(f"Saving model: {output_filename}")
    model.save(output_filename)

    # load and test model
    if should_verify:
        LOGGER.info("Verifying saved model...")
        saved_model = tf.keras.models.load_model(output_filename)
        saved_model.summary()
        verify(model_name, saved_model, image_url, text_options, verbose=True)

    # Dedicated export of image or text encoder
    if image_output is not None or all:
        image_output_filename = get_cache_path(model_name, image_output) if image_output else get_cache_path(model_name,
                                                                                                             output,
                                                                                                             "image")
        LOGGER.info(f"Saving image encoder model: {image_output_filename}")
        model.visual.save(image_output_filename)

    text_output = text_output or (output.format(model="text_{model}") if all else None)
    if text_output is not None:
        text_output_filename = get_cache_path(model_name, text_output) if text_output else get_cache_path(model_name,
                                                                                                          output,
                                                                                                          "text")
        LOGGER.info(f"Saving text encoder model: {text_output_filename}")
        inputs = keras.Input(shape=(None,), name="text", dtype=tf.int32)

        # we have to create a layer to capture all variables used inside of encode_text as well. TODO: more elegant solution
        class TextEncoder(tf.keras.layers.Layer):
            def __init__(self, model: tf.keras.models.Model):
                super().__init__()
                self.model = model
            def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
                return model.encode_text(inputs)

        outputs = TextEncoder(model)(inputs)
        text_encoder = keras.models.Model(inputs=inputs, outputs=outputs)
        text_encoder.save(text_output_filename)
