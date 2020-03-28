import nnhealpix
from nnhealpix.layers import ConvNeighbours, MaxPooling
import keras
from keras.layers import Dense, Dropout, BatchNormalization, Reshape, Flatten
import numpy as np
import sys

def assert_valid_nside(nside):
    sqnside = np.sqrt(nside)
    if np.round(sqnside) != sqnside:
        raise ValueError('Invalid nside = ' + str(nside))

custom_objects={
        'OrderMap': nnhealpix.layers.OrderMap
     }

def create_model(input_dim, nside_min = 32, inner_layer_sizes = [],
                 n_filters=32, pretrained='', l2=0, dropout_rate=0.,
                 normalize=False,  activation='relu', output_activation = 'sigmoid',
                 loss='binary_crossentropy', metrics='accuracy'):
    nside = np.sqrt(input_dim / 12)
    if np.round(nside) != nside:
        raise ValueError('unexpected input_dim for cnn_healpix.create_model')

    nside = int(np.round(nside))

    if pretrained:
        try:
            model = keras.models.load_model(pretrained, compile=False, custom_objects=custom_objects)
        except TypeError as ex:
            if 'dict' in str(ex):
                print('Please apply the following patch to NNhealpix/nnhealpix/layers/__init__.py:')
                print('\tcd ~/git/NNhealpix/nnhealpix/layers')
                print('\tpatch < ~/git/uhecr_aniso/src/nnhealpix_layers.patch')
                exit(1)
            else:
                raise sys.exc_info()  # pass the exception
    else:
        input = keras.Input((input_dim,))
        x = input
        reg = keras.regularizers.l2(l2)
        if normalize:
            x = BatchNormalization()(x)

        do_conv = (nside>nside_min)

        if do_conv:
            x = Reshape((-1, 1))(x)

        while nside>nside_min:
            if dropout_rate>0:
                x = Dropout(rate=dropout_rate)(x)

            x = ConvNeighbours(nside, filters=n_filters, kernel_size=9)(x)
            x = keras.layers.Activation('relu')(x)
            x = MaxPooling(nside, nside // 2)(x)
            nside = nside // 2

        if do_conv:
            x = Flatten()(x)

        for size in inner_layer_sizes:
            if dropout_rate > 0:
                x = Dropout(rate=dropout_rate)(x)
            x = Dense(size, activation=activation, kernel_regularizer=reg)(x)

        out = Dense(1, activation=output_activation, kernel_regularizer=reg)(x)

        model = keras.Model(inputs=[input], outputs=[out])

    optimizer = keras.optimizers.Adadelta()
    model.compile(loss=loss,
              optimizer=optimizer, metrics=[metrics])

    return model
