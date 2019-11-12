# -*- coding:utf-8 -*-

import sys
import tensorflow as tf
from tensorflow.python.keras.initializers import Zeros
from tensorflow.python.keras.layers import Layer

def activation_layer(activation):
    if (isinstance(activation, str)) or (sys.version_info.major == 2 and isinstance(activation, (str, unicode))):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer