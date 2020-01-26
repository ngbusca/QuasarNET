from __future__ import print_function
import numpy as np

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D, concatenate, Lambda
from keras.models import Model
import keras.backend as K
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform, glorot_uniform
from keras import regularizers
from keras.activations import softmax, relu
from functools import partial
import tensorflow as tf

def QuasarNET(input_shape =  None, boxes = 13, nlines = 1, reg_conv = 0., reg_fc=0):
    
    X_input = Input(input_shape)
    X = X_input

    nlayers=4
    nfilters_max = 100
    filter_size=10
    strides = 2
    for stage in range(nlayers):
        nfilters = 100
        X = Conv1D(nfilters, filter_size, strides = strides,
                name = 'conv_{}'.format(stage+1),
                kernel_initializer=glorot_uniform(),
                kernel_regularizer=regularizers.l2(reg_conv))(X)
        X = BatchNormalization(axis=-1)(X)
        X = Activation('relu')(X)

    X = Flatten()(X)
    X = Dense(nfilters_max, activation='linear', name='fc_common')(X)
    X = BatchNormalization()(X)
    X = Activation('relu', name='fc_activation')(X)

    outputs = []
    X_box = []
    for i in range(nlines):
        X_box_aux = Dense(boxes, activation='sigmoid', 
                name='fc_box_{}'.format(i), 
                kernel_initializer=glorot_uniform())(X)
        #X_offset_aux = Dense(boxes, activation='sigmoid',
        X_offset_aux = Dense(boxes, activation='linear',
                name='fc_offset_{}'.format(i), 
                kernel_initializer=glorot_uniform())(X)
        ## rescale the offsets to output between -0.1 and 1.1
        X_offset_aux = Lambda(lambda x:-0.1+1.2*x)(X_offset_aux)
        X_box_aux = concatenate([X_box_aux, X_offset_aux], 
                name="conc_box_{}".format(i))
        X_box.append(X_box_aux)
    
    for b in X_box:
        outputs.append(b)

    model = Model(inputs=X_input, outputs=outputs, name='QuasarNET')

    return model

def custom_loss(y_true, y_pred):
    assert y_pred.shape[1]%2 == 0
    nboxes = y_pred.get_shape().as_list()[1]//2

    N1 = tf.reduce_sum(y_true[...,0:nboxes]) + K.epsilon()
    N2 = tf.reduce_sum((1-y_true[...,0:nboxes])) + K.epsilon()
    loss_class = -tf.reduce_sum(y_true[...,0:nboxes]*tf.math.log(K.clip(y_pred[...,0:nboxes], K.epsilon(), 1-K.epsilon())))/N1
    loss_class -= tf.reduce_sum((1-y_true[...,0:nboxes])*tf.math.log(K.clip(1-y_pred[...,0:nboxes], K.epsilon(), 1-K.epsilon())))/N2

    offset_true = y_true[...,nboxes:]
    offset_pred = y_pred[...,nboxes:]
    doffset = tf.subtract(offset_true, offset_pred)
    loss_offset = tf.reduce_sum(y_true[...,0:nboxes]*tf.square(doffset))/N1

    return tf.add(loss_class, loss_offset)
