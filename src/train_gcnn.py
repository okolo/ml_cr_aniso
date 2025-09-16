import os
import time
import matplotlib

from os import path, remove
from pathlib import Path
from sys import stdout
from datetime import datetime

from tensorflow import keras
from keras.models import Model

from models import create_model

from beta import calc_detectable_frac
from train import (
    source_data, train_seed, val_seed, test_seed,
    add_arg, cl_args,  get_loss,
    init_train_cline_args,
    plot_learning_curves
)
from generators import SampleGeneratorKeras as SampleGenerator


def main():
    init_train_cline_args('Train dynamic graph convolutional network')
    add_arg('--disable_dinamic_conv', action='store_true', help='disable dynamic convolutions (use standard graph convolutions)')
    add_arg('--use_energy_as_feature', action='store_true', help='use energy as feature only (not coordinate)')

    args = cl_args()

    if args.source_id == 'all':
        args.source_id = ','.join(sorted(source_data.keys()))

    sources = sorted(args.source_id.split(','))

    if not args.show_fig:
        matplotlib.use('Agg')  # enable figure generation without running X server session

    args.loss = get_loss(args.loss)

    train_gen = SampleGenerator(args=args, seed=train_seed)
    val_gen = SampleGenerator(args=args, deterministic=True, seed=val_seed, n_samples=args.n_validation_samples, mf=args.validation_mf)
    test_gen = [SampleGenerator(args=args, deterministic=True, seed=test_seed, n_samples=args.n_test_samples)]

    if len(sources) > 1:
        test_gen += [SampleGenerator(args=args, deterministic=True, seed=test_seed,
                                     n_samples=args.n_test_samples, sources=[s]) for s in sources]

    test_b_gen = [SampleGenerator(args=args, deterministic=True, seed=test_seed, n_samples=args.n_test_samples,
                                      mf=args.test_mf)]
    if len(sources) > 1:
        test_b_gen += [SampleGenerator(args=args, deterministic=True, seed=test_seed, n_samples=args.n_test_samples,
                                       sources=[s], mf=args.test_mf) for s in sources]

    results_dir = Path(os.getcwd()).parent / "results" / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"Making results dirs at {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    def train_model(
            _model: Model,
            save_name: str,
            result_path: Path,
            epochs: int = 400,
            verbose: int | bool = 1,
            n_early_stop_epochs: int = 30,
            batch_size: int = 1024
    ) -> None:

        for i in range(args.min_version, 100000):
            save_path = save_name + "_v" + str(i) + '.h5'
            if not path.isfile(result_path / save_path):
                with open(result_path / save_path, mode='x') as f:
                    pass
                break

        weights_file = '/tmp/' + path.basename(save_path) + '_best.weights.h5'
        weights_file = result_path / weights_file

        frac_log_file = save_path[:-3] + '_det_frac.txt'

        frac_log = open(result_path / frac_log_file, mode='wt', buffering=1)

        print('#epoch\tfracs\talpha', file=frac_log)

        frac_and_alphas = []

        def frac_logging_callback(epoch, logs):
            f_a = calc_detectable_frac(val_gen, model, args)
            frac_and_alphas.append(f_a)

            print(epoch, f_a[0], f_a[1], file=frac_log)
            print('Detectable fraction:', f_a[0], '\talpha =', f_a[1])

            if args.monitor.startswith('frac'):
                f_a_sorted = sorted(frac_and_alphas)

                if len(frac_and_alphas) == 1 or (f_a == f_a_sorted[0] and f_a_sorted[1] != f_a): # last condition added to avoid staying on plato
                    model.save_weights(weights_file, overwrite=True)
                elif n_early_stop_epochs > 0 and len(frac_and_alphas) - frac_and_alphas.index(f_a_sorted[0]) > n_early_stop_epochs:
                    model.stop_training = True
                    print('Early stop on epoch', epoch)

        if args.monitor.startswith('frac'):
            callbacks = [keras.callbacks.LambdaCallback(on_epoch_end=frac_logging_callback)]
        else:
            callbacks = [keras.callbacks.ModelCheckpoint(weights_file, save_best_only=True, monitor=args.monitor,
                                                save_weights_only=True)]  # save best model
            if n_early_stop_epochs > 0:
                callbacks.append(
                    # this doesn't work for 'frac' monitor
                    keras.callbacks.EarlyStopping(
                        monitor=args.monitor, patience=n_early_stop_epochs, verbose=1, min_delta=args.min_delta
                    )
                    # early stop
                )

        validation_data = None if args.monitor.startswith('frac') else val_gen

        t = time.time()

        history = model.fit(train_gen, epochs=epochs, verbose=verbose,
                            validation_data=validation_data, callbacks=callbacks)
        t = time.time() - t

        if n_early_stop_epochs > 0 and path.isfile(weights_file):
            model.load_weights(weights_file)  # load best weights
            # remove(weights_file)

        print('Training took %.0f sec' % t)

        model.save(result_path / save_path)
        print('Model saved in', result_path / save_path)

        if n_early_stop_epochs > 0 and path.isfile(weights_file):
            remove(weights_file)

        if args.plot_learning_curves:
            plot_learning_curves(history, save_file=result_path / (save_path[:-3] + '_train.png'), show_fig=args.show_fig)

        if args.evaluate_test_loss:
            score = model.evaluate_generator(test_gen[0], verbose=0)
        else:
            score = []

        with open(result_path / (save_path + '.args'), mode='w') as args_out:
            from sys import argv

            print(*argv, file=args_out)
            print(file=args_out)

            for key, val in args.__dict__.items():
                print(key, '=', val, file=args_out)

        with open(result_path / (save_path + '.score'), mode='w') as out:
            for name, sc in zip(model.metrics_names, score):
                print(name, sc, file=out)
                print(name, sc)

            print('training_time_sec', t, file=out)

            if len(frac_and_alphas) > 0:
                frac_and_alphas.sort()
                f, a = frac_and_alphas[0]
                for file in [out, stdout]:
                    print('best_val_frac_' + args.validation_mf, f, file=file)
                    print('best_val_alpha_' + args.validation_mf, a, file=file)

            for gen in test_gen:
                sources = gen.sources
                if sources is None:
                    src_name = 'average' if ',' in args.source_id else args.source_id
                else:
                    src_name = ','.join(sources)
                print('testing on', src_name,'source')
                # print(args.mf, 'field..')
                test_frac, test_alpha = calc_detectable_frac(gen, model, args)
                for file in [out, stdout]:
                    print('frac_' + src_name + '_' + args.mf, test_frac, file=file)
                    print('alpha_' + src_name + '_' + args.mf, test_alpha, file=file)
                b_gen = [b for b in test_b_gen if b.sources == gen.sources]
                if len(b_gen) == 1:
                    test_frac, test_alpha = calc_detectable_frac(b_gen[0], model, args)
                    for file in [out, stdout]:
                        print('frac_' + src_name + '_' + args.test_mf, test_frac, file=file)
                        print('alpha_' + src_name + '_' + args.test_mf, test_alpha, file=file)

    n_features = 3 if args.exclude_energy else 4
    if args.use_energy_as_feature and not args.exclude_energy:
        n_coords = n_features - 1
    else:
        n_coords = n_features

    model = create_model(args.Neecr, n_coords=n_coords, n_features=n_features, pretrained=args.pretrained,
                         dinamic_conv=(not args.disable_dinamic_conv))

    if args.pretrained and len(args.output_prefix) == 0:
        save_name = args.pretrained[:-3]
    else:
        if args.exposure == 'uniform':
            prefix = ''
        else:
            prefix = args.exposure + '_'
        prefix += '_'.join(sorted(args.source_id.split(',')))
        save_name = args.output_prefix + f'{prefix}_N{args.Neecr}_B{args.mf}'
        if args.threshold > 0:
            save_name += '_th' + str(args.threshold)
        save_name += '_sig{:.0f}'.format(100*args.sigmaLnE)
        if args.exclude_energy:
            save_name += '_noE'
        elif args.use_energy_as_feature:
            save_name += '_noEcoord'

    train_model(
        model, save_name,
        result_path=results_dir,
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        n_early_stop_epochs=args.n_early_stop
    )


if __name__ == '__main__':
    main()
