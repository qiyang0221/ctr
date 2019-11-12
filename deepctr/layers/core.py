# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.keras.keras import backend as K
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2

from .activation import activation_layer

class DNN(Layer):
    ''' The Multi Layer Percetron
        Input shape
            - nD tensor with shape
        
        Output shape
            - nD tensor with shape
        
        Arguments
            - **hidden_units**: list of positve integer, the layer number and units in each layer
            - **activation**: Activation function to use
            - **l2**: float between 0 and 1. L2 regularizers strength applied to the kernel weights matrix.
            - **dropout_rate**: float in [0, 1). Fraction of the units to dropout
            - **use_bn**: bool. Whether use BatchNormalization before activation or not
            - **seed**: random seed
    '''

    def __init__(self, hidden_units, activation='relu', l2=0, dropout_rate=0, use_bn=False, seed=1024, *args, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2 = l2
        self.use_bn = use_bn
        super().__init__(*args, **kwargs)
    
    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='keneral' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2),
                                        trainable=True) for i in range(len(self.hidden_units))]