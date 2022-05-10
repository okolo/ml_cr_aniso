from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAvgPool1D, Input, Dropout, BatchNormalization

import edgeconv

def layer_name(prefix):
    return edgeconv.EdgeConv.layer_name(prefix)

custom_objects = {"EdgeConv": edgeconv.EdgeConv, "SplitLayer": edgeconv.SplitLayer, "mean": keras.backend.mean}

def create_model(n_points, n_features=4, kernel_layers=5*[52], n_conv=5, dense_layer_sizes=[380], k_neighbors=16,
                 pretrained='', l2=0, l1=0, dropout_rate=0.1,
                 normalize=True,  activation='prelu', output_activation = 'sigmoid',
                 loss='binary_crossentropy', metrics='accuracy', lr=0.001):
    inline_activation = activation
    layer_activation = None
    if activation == 'prelu':
        inline_activation = 'linear'
        def layer_activation():
            return keras.layers.PReLU(name=layer_name('prelu'))

    weight_reg = None
    if l1 > 0 or l2 > 0:
        weight_reg = keras.regularizers.L1L2(l1=l1, l2=l2)

    if pretrained:
        model = keras.models.load_model(pretrained, compile=False, custom_objects=custom_objects)
    else:
        points = Input((n_points, n_features), name=layer_name('dgcnn_in'))
        features = points
        norm_features = points
        if normalize:
            norm_features = BatchNormalization(name=layer_name('batch_norm'))(features)

        def EdgeConv(x):
            return edgeconv.EdgeConv(next_neighbors=k_neighbors,
                 kernel_layers=kernel_layers,
                 kernel_l1=l1,
                 kernel_l2=l2,
                 kernel_activation=activation)(x)

        out = EdgeConv([points, norm_features])
        for i in range(1, n_conv):
            out = EdgeConv(out)
        out = GlobalAvgPool1D()(out)
        for dim in dense_layer_sizes:
            out = Dense(dim, activation=inline_activation, kernel_regularizer=weight_reg, name=layer_name('dense'))(out)
            if layer_activation is not None:
                out = layer_activation()(out)
            if dropout_rate > 0:
                out = Dropout(dropout_rate, name=layer_name('dropout'))(out)
        out = Dense(1, activation=output_activation, kernel_regularizer=weight_reg, name=layer_name('dense'))(out)
        model = keras.models.Model([points], out)

    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=loss,
              optimizer=optimizer, metrics=[metrics])

    # tmp_model_file = '/tmp/save_test.h5'
    # model.save(tmp_model_file, overwrite=True)
    # model = keras.models.load_model(tmp_model_file, compile=True, custom_objects=custom_objects)

    return model
