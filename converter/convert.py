import hashlib
import os
import re
import sys
import urllib
import warnings
from typing import List

import numpy as np
import requests
import torch.hub

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

import clip
from PIL import Image


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

    # print("Unmapped keys:")
    # unmapped_keys = set(state_dict.keys()).difference(mapped_keys)
    # for k in unmapped_keys:
    #     print(f"missing '{k}' -> '?'")
    # exit(0)


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
    text = text.detach().numpy()
    logits_per_image, logits_per_text = keras_model.predict((image, text))
    tf_probs = tf.nn.softmax(logits_per_image, axis=1)
    tf_probs = np.array(tf_probs)

    if verbose:
        print(f"Classify image: {image_url}")
        print(f"Text options: {text_options}")
        print(f"Pytorch: {torch_probs}")
        print(f"Tensorflow: {tf_probs}")

    assert np.abs(torch_probs - tf_probs).sum() < 1e-3, f"PyTorch and Tensorflow results should be almost equal: torch_probs={torch_probs}, tf_probs={tf_probs}"
