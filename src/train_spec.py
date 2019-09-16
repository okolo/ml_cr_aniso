import numpy as np
import argparse
import losses

norm_file = 'samples/ALM_rand_f/aps_CenA_D3.5_Emin56_Neecr500_Nsample3000_R1_Nside512_100.npz'

feature_names = ['spectrum']

def load_features(npz_file):
    features_to_use = sorted(list(set(feature_names)))
    npz = np.load(npz_file)
    fractions = npz['fractions']
    absent_features = [feature_name for feature_name in features_to_use if feature_name not in npz]
    if len(absent_features) > 0:
        raise ValueError("Feature(s) '" + "','".join(absent_features) + "' are absent in " + npz_file)

    features = [npz[f] for f in features_to_use]
    empty_features = [feature_name for feature_name, f in zip(features_to_use,features) if len(f.shape) == 0]
    if len(empty_features) > 0:
        raise ValueError("Feature(s) '" + "','".join(empty_features) + "' are empty in " + npz_file)

    if len(features) > 1:
        features = np.hstack(features)
    else:
        features = features[0]
    return fractions, features

def normalize_features(features):
    _, norm_spec = load_features(norm_file)
    norm_mean = np.mean(norm_spec, axis=0, keepdims=True)
    norm_std = np.std(norm_spec, axis=0, keepdims=True)
    return (features - norm_mean) / (norm_std + 1e-15)

def get_loss(loss):
    # try to find loss by name in losses module
    # if not found use string as is
    try:
        loss = getattr(losses, loss)
    except AttributeError:
        pass
    return loss

if __name__ == '__main__':

    import time
    import matplotlib
    from os import path, remove
    import glob

    cline_parser = argparse.ArgumentParser(description='Train network')
    def add_arg(*pargs, **kwargs):
        cline_parser.add_argument(*pargs, **kwargs)


    add_arg('data', type=str, nargs='*', metavar='npz', help='npz data file(s) or file mask(s)', default=["*.npz"])
    add_arg('--loss', type=str, help='loss function (default mse/binary_crossentropy)', default='')
    add_arg('--metrics', type=str, help='metrics function (default mae/accuracy)', default='')
    add_arg('--threshold', type=float, help='source fraction threshold for binary classification (if zero, regression mode is used)', default=0.01)
    add_arg('--layers', type=str, help='comma-separated list of inner layer sizes', default='')
    add_arg('--output_prefix', type=str, help='output model file path prefix', default='')
    add_arg('--batch_size', type=int, help='size of training batch', default=10000)
    add_arg('--n_epochs', type=int, help='number of training epochs', default=100000)
    add_arg('--show_fig', action='store_true', help="Show learning curves")
    add_arg('--n_early_stop', type=int, help='number of epochs to monitor for early stop', default=30)
    add_arg('--pretrained', type=str, help='pretrained network', default='')
    add_arg('--features', type=str, nargs='*', metavar='F', help='features to use (currently available spectrum and/or alm)', default=feature_names)
    add_arg('--weights', action='store_true', help="Balance set using weights (works for classification mode only)")


    args = cline_parser.parse_args()

    if not args.show_fig:
        matplotlib.use('Agg')  # enable figure generation without running X server session

    feature_names = args.features

    import matplotlib.pyplot as plt

    inner_layers = [int(l) for l in args.layers.split(',') if len(l) > 0]

    if not args.loss:
        args.loss = 'binary_crossentropy' if args.threshold > 0 else "mse"
    else:
        args.loss = get_loss(args.loss)
    if not args.metrics:
        args.metrics = 'accuracy' if args.threshold > 0 else "mean_absolute_error"

    def load_npz():
        for mask in args.data:
            for f in glob.glob(mask):
                try:
                    yield load_features(f)
                except ValueError as er:
                    print(str(er))

    data = list(load_npz())
    if len(data) == 0:
        print("no valid input files found matching the mask(s) '" + "', '".join(args.data) + "'")
        exit(1)

    fractions, features = zip(*data)

    fractions = np.hstack(fractions)
    features = np.vstack(features)

    #
    #
    # spectrum = [npz['spectrum'] for npz in npzs]
    #
    # features = [[npz[f] for f in features_to_use] for npz in npzs]
    #
    # if len(npzs) == 1:
    #     spectrum = spectrum[0]
    #     fractions = fractions[0]
    # else:
    #     spectrum = np.vstack(spectrum)
    #     fractions = np.hstack(fractions)

    features = normalize_features(features)

    xy = np.hstack((features, fractions.reshape(-1, 1)))
    np.random.shuffle(xy)

    x = xy[:, :-1]
    y = xy[:, -1]
    weights = None

    if args.threshold > 0:
        y = (y > args.threshold).astype(np.float32)
        output_activation = 'sigmoid'
        if args.weights:
            frac1 = np.sum(y)/len(y)
            assert frac1 > 0, 'no samples above threshold'
            assert frac1 < 1, 'no samples below threshold'
            weights = np.ones_like(y)*(0.5/frac1)
            weights[y == 0] = 0.5/(1-frac1)


    else:
        output_activation = 'linear'

    # split data to train, test and validation parts
    n_test = int(np.round(0.1 * len(fractions)))

    x_test = x[-n_test:]
    y_test = y[-n_test:]
    x_val = x[-2*n_test:-n_test]
    y_val = y[-2*n_test:-n_test]
    x_train = x[:-2*n_test]
    y_train = y[:-2*n_test]

    if weights is not None:
        weights_test = weights[-n_test:]
        weights_val = weights[-2 * n_test:-n_test]
        weights_train = weights[:-2 * n_test]
        validation_data = (x_val, y_val, weights_val)
    else:
        weights_test = weights_val = weights_train = None
        validation_data = (x_val, y_val)

    ################### Create Model in Keras

    import keras
    from keras.layers import Dense, Dropout, BatchNormalization

    def create_model(input_dim, inner_layer_sizes, pretrained='', l2=0, dropout_rate=0., normalize=False, activation='relu'):
        if pretrained:
            model = keras.models.load_model(pretrained, compile=False)
        else:
            input = keras.Input((input_dim,))
            x = input
            reg = keras.regularizers.l2(l2)
            if normalize:
                x = BatchNormalization()(x)
            for size in inner_layer_sizes:
                if dropout_rate>0:
                    x = Dropout(rate=dropout_rate)(x)
                x = Dense(size, activation=activation, kernel_regularizer=reg)(x)

            out = Dense(1, activation=output_activation, kernel_regularizer=reg)(x)

            model = keras.Model(inputs=[input], outputs=[out])

        optimizer = keras.optimizers.Adadelta()
        model.compile(loss=args.loss,
                  optimizer=optimizer, metrics=[args.metrics])
        return model


    def plot_learning_curves(history, save_file=None):
        metrics = ['loss'] + [m for m in history.history if m != 'loss' and not m.startswith('val_')]

        for i, m in enumerate(metrics):
            plt.subplot(len(metrics), 1, 1+i)
            plt.plot(history.history['val_' + m], 'r')
            plt.plot(history.history[m], 'g')
            plt.legend(['Test', 'Train'], loc='upper right')
            plt.ylabel(m)

        if save_file:
            plt.savefig(save_file)

        try:
            if args.show_fig:
                plt.show()
        except Exception:
            pass


    def train_model(model, save_name, epochs=400, verbose=1, n_early_stop_epochs=30, batch_size=1024):
        for i in range(100000):
            save_path = save_name + "_v" + str(i) + '.h5'
            if not path.isfile(save_path):
                with open(save_path, mode='x') as f:
                    pass
                break

        weights_file = '/tmp/' + path.basename(save_path) + '_best_weights.h5'

        callbacks = []
        if n_early_stop_epochs > 0:
            callbacks = [
                keras.callbacks.ModelCheckpoint(weights_file, save_best_only=True, save_weights_only=True),
                # save best model
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=n_early_stop_epochs, verbose=1)  # early stop
            ]
        t = time.time()

        history = model.fit(x_train, y_train, sample_weight=weights_train,
                            batch_size=batch_size, epochs=epochs, verbose=verbose,
                            validation_data=validation_data,
                            callbacks=callbacks)
        t = time.time() - t
        if n_early_stop_epochs > 0 and path.isfile(weights_file):
            model.load_weights(weights_file)  # load best weights
            remove(weights_file)

        print('Training took %.0f sec' % t)
        model.save(save_path)
        print('Model saved in', save_path)

        plot_learning_curves(history, save_file=save_path[:-3] + '_train.png')

        score = model.evaluate(x_test, y_test, verbose=0, sample_weight=weights_test)

        with open(save_path + '.score', mode='w') as out:
            for name, sc in zip(model.metrics_names, score):
                print(name, sc, file=out)
                print(name, sc)
            print('training_time_sec', t, file=out)

    model = create_model(x_train.shape[1], inner_layers, pretrained=args.pretrained)
    if args.pretrained and len(args.output_prefix) == 0:
        save_name = args.pretrained[:-3]
    else:
        save_name = args.output_prefix + '_'.join(args.features) + "_L" + '_'.join([str(i) for i in [x_train.shape[1]] + inner_layers]) + "_th" + str(args.threshold)

    train_model(model, save_name, batch_size=args.batch_size, epochs=args.n_epochs, n_early_stop_epochs=args.n_early_stop)

