import numpy as np

from keras.losses import mse
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv1D, AveragePooling1D, MaxPooling1D, concatenate, Reshape, Permute
from keras.models import Model, load_model, save_model
import keras.backend as K
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform, glorot_uniform
from keras import regularizers
from keras.activations import softmax, relu
from functools import partial
import tensorflow as tf

def QuasarNET(input_shape =  None, classes = 6, boxes = 13, nlines = 1, reg_conv = 0., reg_fc=0):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = X_input

    nlayers=4
    nfilters_max = 100
    filter_size=10
    strides = 2
    for stage in range(nlayers):
        nfilters = 100
        print X.shape
        X = Conv1D(nfilters, filter_size, strides = strides, name = 'conv{}'.format(stage+1), kernel_initializer = glorot_uniform(), kernel_regularizer=regularizers.l2(reg_conv))(X)
        X = BatchNormalization(axis=-1)(X)
        X = Activation('relu')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(nfilters_max, activation='linear', name='fc_common')(X)
    X = BatchNormalization()(X)
    X = Activation('relu', name='fc_activation')(X)

    X_class = Dense(classes, activation='softmax', name='fc_softmax' + str(classes), kernel_initializer = glorot_uniform())(X)
    X_bal = Dense(1, activation='sigmoid', name='fc_bal', kernel_initializer = glorot_uniform())(X)

    outputs = [X_class, X_bal]
    X_box = []
    for i in range(nlines):
        X_box_aux = Dense(boxes, activation='softmax', name='fc_box_{}'.format(i), kernel_initializer = glorot_uniform())(X)
        X_offset_aux = Dense(boxes, activation='sigmoid', name='fc_offset_{}'.format(i), kernel_initializer = glorot_uniform(), kernel_regularizer=regularizers.l2(reg_fc))(X)
        X_box_aux = concatenate([X_box_aux, X_offset_aux], name="conc_box_{}".format(i))
        print "adding additional lines", X_box_aux.shape
        X_box.append(X_box_aux)
    
    for b in X_box:
        outputs.append(b)

    if len(X_box)==1:
        Z = Dense(1, activation='relu', name="redshift", kernel_initializer = glorot_uniform())(X_box[0])
        outputs.append(Z)
    elif len(X_box)>1:
        Z = Dense(1, activation='relu', name="redshift", kernel_initializer = glorot_uniform())(concatenate(X_box))
        outputs.append(Z)

    # Create model
    model = Model(inputs = X_input, outputs = outputs, name='QuasarNET')

    return model

def custom_loss(y_true, y_pred):
    assert y_pred.shape[1]%2 == 0
    nboxes = y_pred.shape[1]/2
    loss_class = K.categorical_crossentropy(y_true[...,0:nboxes], y_pred[...,0:nboxes])
    offset_true = y_true[...,nboxes:]
    offset_pred = y_pred[...,nboxes:]
    doffset = tf.subtract(offset_true, offset_pred)
    doffset = tf.where(y_true[...,0:nboxes]==1, doffset, y_true[...,0:nboxes])
    loss_offset = K.mean(tf.square(doffset))

    return tf.add(loss_class, loss_offset)
