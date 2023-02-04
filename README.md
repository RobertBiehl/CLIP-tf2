# CLIP-tf2
OpenAI CLIP converted to Tensorflow 2/Keras

__Official Repository__: https://github.com/openai/CLIP

## Model conversion
```sh
$ python convert_clip.py --help

       USAGE: convert_clip.py [flags]
flags:

convert_clip.py:
  --[no]all: Export all versions. (will use output location if image_output or
    text_output are not present)
    (default: 'false')
  --image_output: Image encoder Keras SavedModel output destination (optional)
  --model: <RN50|RN101|RN50x4|ViT-B/32>: CLIP model architecture to convert
    (default: 'RN50')
  --output: CLIP Keras SavedModel Output destination
    (default: 'models/CLIP_{model}')
  --text_output: Text encoder Keras SavedModel output destination (optional)
```

Example:
```sh
$ python convert_clip.py --model RN50 --output models/CLIP_{model}
```
Output: 
```sh
Copying weights: 100%|██████████| 482/482 [00:00<00:00, 674.13it/s]
I0523 18:18:40.867926 4600192512 builder_impl.py:774] Assets written to: CLIP_RN50/assets

Model: "clip"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
visual (ModifiedResNet)      multiple                  38370144  
_________________________________________________________________
transformer (Transformer)    multiple                  37828608  
_________________________________________________________________
ln_final (LayerNorm)         multiple                  1024      
=================================================================
Total params: 102,060,385
Trainable params: 102,007,137
Non-trainable params: 53,248
_________________________________________________________________
Classify image: https://github.com/openai/CLIP/blob/main/CLIP.png?raw=true
Text options: ['a diagram', 'a dog', 'a cat', 'a neural network']
Pytorch: [[0.24351287 0.00320374 0.00082513 0.7524583 ]]
Tensorflow: [[0.24351244 0.00320391 0.0008252  0.7524584 ]]

Process finished with exit code 0
```

__Exporting standalone encoders:__

Image encoder:
```sh
$ python convert_clip.py --model RN50 --image_output models/CLIP_image_{model}
```

Text encoder:
```sh
$ python convert_clip.py --model RN50 --text_output models/CLIP_image_{model}
```

## Currently supported models:
- RN50
- RN101
- RN50x4
- RN50x16
- RN50x64
- ViT-B/32
- ViT-B/16
- ViT-L/14
- ViT-L/14@336px

## Tasks
- [x] Convert PyTorch to Tensorflow model (RN)
- [x] Export as Tensorflow SavedModel
- [x] ViT conversion
- [x] Export standalone image and text encoders
- [x] Installable pip package
- [ ] Improve API: loading model, usage
- [ ] Float16 support
- [ ] Make PyTorch dependency optional (only for updating model from official weights)
- [ ] Implement training


