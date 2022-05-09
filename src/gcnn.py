from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAvgPool1D, Flatten, Subtract, Concatenate, Input, Dropout, BatchNormalization

import edgeconv

layer_idx = [-1]

def layer_name(prefix):
    layer_idx[0] += 1
    return prefix + f'_{layer_idx[0]}'

custom_objects = {"EdgeConv": edgeconv.EdgeConv, "SplitLayer": edgeconv.SplitLayer, "mean": keras.backend.mean}

def create_model(n_points, n_features=4, kernel_layers=[30, 20], n_conv=3, dense_layer_sizes=[256], k_neighbors=10,
                 pretrained='', l2=0, l1=0, dropout_rate=0.,
                 normalize=False,  activation='relu', output_activation = 'sigmoid',
                 loss='binary_crossentropy', metrics='accuracy', lr=0.001):
    weight_reg = None
    if l1 > 0 or l2 > 0:
        weight_reg = keras.regularizers.L1L2(l1=l1, l2=l2)
    # def kernel_func(data):
    #     d1, d2 = data
    #     dif = Subtract(name=layer_name('subtract'))([d1, d2])
    #     x = Concatenate(axis=-1, name=layer_name('concat'))([d1, dif])
    #     for dim in kernel_layers:
    #         x = Dense(dim, activation=activation, kernel_regularizer=weight_reg, name=layer_name('kern_dense'))(x)
    #     return x

    if pretrained:
        model = keras.models.load_model(pretrained, compile=False, custom_objects=custom_objects)
    else:
        points = Input((n_points, n_features), name=layer_name('dgcnn_in'))
        features = points
        #features = Input((n_points, n_features))
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
            out = Dense(dim, activation=activation, kernel_regularizer=weight_reg, name=layer_name('dense'))(out)
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