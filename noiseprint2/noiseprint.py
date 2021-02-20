from PIL import Image
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation

import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model

from noiseprint2.utility import jpeg_quality_of_file


# Bias layer necessary because noiseprint applies bias after batch-normalization.
class BiasLayer(tf.keras.layers.Layer):
    """
    Simple bias layer
    """

    def build(self, input_shape):
        self.bias = self.add_weight('bias', shape=input_shape[-1], initializer="zeros")

    @tf.function
    def call(self, inputs, training=None):
        return inputs + self.bias


def _full_conv_net(num_levels=17, padding='SAME'):
    """FullConvNet model."""
    activation_fun = [tf.nn.relu, ] * (num_levels - 1) + [tf.identity, ]
    filters_num = [64, ] * (num_levels - 1) + [1, ]
    batch_norm = [False, ] + [True, ] * (num_levels - 2) + [False, ]

    inp = tf.keras.layers.Input([None, None, 1])
    model = inp

    for i in range(num_levels):
        model = Conv2D(filters_num[i], 3, padding=padding, use_bias=False)(model)
        if batch_norm[i]:
            model = BatchNormalization(epsilon=1e-5)(model)
        model = BiasLayer()(model)
        model = Activation(activation_fun[i])(model)

    return Model(inp, model)


def setup_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)


class NoiseprintEngine:
    save_path = os.path.join(os.path.dirname(__file__), './weights/net_jpg%d/')
    slide = 1024  # 3072
    large_limit = 1050000  # 9437184
    overlap = 34

    def __init__(self):
        self._model = _full_conv_net()
        self._loaded_quality = None

    def load_quality(self, quality):
        """
        Loads a quality level for the next noiseprint predictions.
        Quality level can be obtained by the file quantization table.
        :param quality: Quality level, int between 51 and 101 (included)
        """
        if quality < 51 or quality > 101:
            raise ValueError("Quality must be between 51 and 101 (included). Provided quality: %d" % quality)
        if quality == self._loaded_quality:
            return
        checkpoint = self.save_path % quality
        self._model.load_weights(checkpoint)
        self._loaded_quality = quality

    @tf.function(experimental_relax_shapes=True,
                 input_signature=[tf.TensorSpec(shape=(1, None, None, 1), dtype=tf.float32)])
    def _predict(self, img):
        return self._model(img)

    def _predict_small(self, img):
        return np.squeeze(self._predict(img[np.newaxis, :, :, np.newaxis]).numpy())

    def _predict_large(self, img):
        # prepare output array
        res = np.zeros((img.shape[0], img.shape[1]), np.float32)

        # iterate over x and y, strides = self.slide, window size = self.slide+2*self.overlap
        for x in range(0, img.shape[0], self.slide):
            x_start = x - self.overlap
            x_end = x + self.slide + self.overlap
            for y in range(0, img.shape[1], self.slide):
                y_start = y - self.overlap
                y_end = y + self.slide + self.overlap
                patch = img[max(x_start, 0): min(x_end, img.shape[0]), max(y_start, 0): min(y_end, img.shape[1])]
                patch_res = np.squeeze(self._predict_small(patch))

                # discard initial overlap if not the row or first column
                if x > 0:
                    patch_res = patch_res[self.overlap:, :]
                if y > 0:
                    patch_res = patch_res[:, self.overlap:]
                # discard data beyond image size
                patch_res = patch_res[:min(self.slide, patch_res.shape[0]), :min(self.slide, patch_res.shape[1])]
                # copy data to output buffer
                res[x: min(x + self.slide, res.shape[0]), y: min(y + self.slide, res.shape[1])] = patch_res
        return res

    def predict(self, img):
        """
        Run the noiseprint generation CNN over the input image
        :param img: input image, 2-D numpy array
        :return: output noisepritn, 2-D numpy array with the same size of the input image
        """
        if len(img.shape) != 2:
            raise ValueError("Input image must be 2-dimensional. Passed shape: %r" % img.shape)
        if self._loaded_quality is None:
            raise RuntimeError("The engine quality has not been specified, please call load_quality first")
        if img.shape[0] * img.shape[1] > self.large_limit:
            return self._predict_large(img)
        else:
            return self._predict_small(img)


def gen_noiseprint(image, quality=None):
    """
    Generates the noiseprint of an image
    :param image: image data. Numpy 2-D array or path string of the image
    :param quality: Desired quality level for the noiseprint computation.
    If not specified the level is extracted from the file if image is a path string to a JPEG file, else 101.
    :return: The noiseprint of the input image
    """
    if isinstance(image, str):
        # if image is a path string, load the image and quality if not defined
        if quality is None:
            try:
                quality = jpeg_quality_of_file(image)
            except AttributeError:
                quality = 101
        image = np.asarray(Image.open(image).convert("YCbCr"))[..., 0].astype(np.float32) / 256.0
    else:
        if quality is None:
            quality = 101
    engine = NoiseprintEngine()
    engine.load_quality(quality)
    return engine.predict(image)


def normalize_noiseprint(noiseprint, margin=34):
    """
    Normalize the noiseprint between 0 and 1, in respect to the central area
    :param noiseprint: noiseprint data, 2-D numpy array
    :param margin: margin size defining the central area, default to the overlap size 34
    :return: the normalized noiseprint data, 2-D numpy array with the same size of the input noiseprint data
    """
    v_min = np.min(noiseprint[margin:-margin, margin:-margin])
    v_max = np.max(noiseprint[margin:-margin, margin:-margin])
    return ((noiseprint - v_min) / (v_max - v_min)).clip(0, 1)
