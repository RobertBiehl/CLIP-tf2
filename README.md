# CLIP-tf2
OpenAI CLIP converted to Tensorflow 2/Keras

__Official Repository__: https://github.com/openai/CLIP

## Model conversion
```
python convert_clip.py --model RN50 --output SAVE/MODEL/HERE/modelname_{model}
```

## Tasks
- [x] Convert PyTorch to Tensorflow model (RN)
- [x] Export as Tensorflow SavedModel
- [ ] Export standalone image and text encoders
- [ ] Float16 support
- [ ] ViT conversion
- [ ] Make PyTorch dependency optional (only for updating model from official weights)
- [ ] Implement training


