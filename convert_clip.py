from absl import app, flags
from clip_tf import converter


FLAGS = flags.FLAGS
flags.DEFINE_enum('model', 'RN50', converter.MODELS.keys(), 'CLIP model architecture to convert')
flags.DEFINE_string('output', 'models/CLIP_{model}', 'CLIP Keras SavedModel Output destination')
flags.DEFINE_string('image_output', None, 'Image encoder Keras SavedModel output destination (optional)')
flags.DEFINE_string('text_output', None, 'Text encoder Keras SavedModel output destination (optional)')
flags.DEFINE_bool('all', False, 'Export all versions. (will use output location if image_output or text_output are not present)')


def main(argv):
    converter.convert(FLAGS.model, FLAGS.output, FLAGS.image_output, FLAGS.text_output, FLAGS.all)


if __name__ == '__main__':
    app.run(main)
