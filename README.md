# CLIP-tf2
OpenAI CLIP converted to Tensorflow 2/Keras

__Official Repository__: https://github.com/openai/CLIP

## Model conversion
```sh
python convert_clip.py --model RN50 --output CLIP_{model}
```
Output: 
```sh
Copying weights: 100%|██████████| 482/482 [00:00<00:00, 674.13it/s]
I0523 18:18:40.867926 4600192512 builder_impl.py:774] Assets written to: CLIP_RN50/assets

Model: "clip_1"
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

## Tasks
- [x] Convert PyTorch to Tensorflow model (RN)
- [x] Export as Tensorflow SavedModel
- [ ] Export standalone image and text encoders
- [ ] Float16 support
- [ ] ViT conversion
- [ ] Make PyTorch dependency optional (only for updating model from official weights)
- [ ] Implement training


