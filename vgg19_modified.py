# -*- coding: utf-8 -*-
"""VGG19 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

"""
from __future__ import print_function
from __future__ import absolute_import

import os
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense,Input,Conv2D,MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils,get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions,preprocess_input,_obtain_input_shape

from modified_conv2d import ModifiedConv2D

def VGG19_Modified(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = ModifiedConv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = ModifiedConv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = ModifiedConv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = ModifiedConv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = ModifiedConv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = ModifiedConv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = ModifiedConv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = ModifiedConv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = ModifiedConv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = ModifiedConv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = ModifiedConv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = ModifiedConv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = ModifiedConv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = ModifiedConv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = ModifiedConv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = ModifiedConv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg19')


    return model
