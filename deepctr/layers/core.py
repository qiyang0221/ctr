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

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='keneral' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]
        
        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed+i) for i in range(len(self.hidden_units))]
        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        super(DNN, self).build(input_shape)
    
    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=trianing)
            
            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training = training)
            deep_input = fc
        
        return deep_input
    
    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape
        
        return tuple(shape)
    
    def get_config(self,):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate, 'seed': self.seed}
        base_config = super(DNN, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
