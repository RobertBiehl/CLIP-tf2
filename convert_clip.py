import numpy as np
from absl import app, flags
import tensorflow as tf

from clip_tf.model import build_model
import converter

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}

FLAGS = flags.FLAGS
flags.DEFINE_enum('model', 'RN50', _MODELS.keys(), 'CLIP model architecture to convert')
flags.DEFINE_string('output', 'models/CLIP_{model}', 'CLIP Keras SavedModel Output destination')
flags.DEFINE_string('image_output', None, 'Image encoder Keras SavedModel output destination (optional)')
flags.DEFINE_string('text_output', None, 'Text encoder Keras SavedModel output destination (optional)')
flags.DEFINE_bool('all', False, 'Export all versions. (will use output location if image_output or text_output are not present)')

# model input for verification
image_url = "https://github.com/openai/CLIP/blob/main/CLIP.png?raw=true"
text_options = ["a diagram", "a dog", "a cat", "a neural network"]


def main(argv):
    sanitized_model_name = FLAGS.model.replace("/", "_")
    model_url = _MODELS[FLAGS.model]
    state_dict = converter.download_statedict(model_url)
    model = build_model(state_dict)

    # predict to build shapes (model.build doesnt work, as it only supports float inputs)
    model.predict((
        np.ones((1, model.image_resolution, model.image_resolution, 3), np.float32),
        np.ones((1, 4, 77), np.int64)
    ))
    converter.load_pytorch_weights(model, state_dict, verbose=False)

    converter.verify(FLAGS.model, model, image_url, text_options, verbose=True)

    # create SavedModel
    output_filename = FLAGS.output.format(model=sanitized_model_name)
    model.save(output_filename)

    # load and test model
    model = tf.keras.models.load_model(output_filename)
    model.summary()
    converter.verify(FLAGS.model, model, image_url, text_options, verbose=True)

    image_output = FLAGS.image_output or (FLAGS.output.format(model="image_{model}") if FLAGS.all else None)
    if image_output is not None:
        image_output_filename = image_output.format(model=sanitized_model_name)
        model.visual.save(image_output_filename)

    text_output = FLAGS.text_output or (FLAGS.output.format(model="text_{model}") if FLAGS.all else None)
    if text_output is not None:
        text_output_filename = text_output.format(model=sanitized_model_name)
        model.visual.save(text_output_filename)



if __name__ == '__main__':
    app.run(main)
