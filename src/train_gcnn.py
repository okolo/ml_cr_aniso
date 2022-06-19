import logging

import argparse
from tensorflow import keras
import time
import matplotlib
from os import path, remove
import healpy as hp
from sys import stderr, stdout
from gcnn import create_model
from astropy.coordinates import SkyCoord
from astropy import units as u
from beta import calc_detectable_frac
from train import *

class SampleGenerator(keras.utils.Sequence):
    def __init__(self, args, deterministic=None, seed=0, n_samples=None, return_frac=False, suffix='*', sources=None,
                 mixture=[], add_iso=None, sampler="auto", batch_size=None, mf=None):
        from exposure import create_exposure
        exposure = create_exposure(args)
        self.point_exposure = []
        self.exposure = exposure
        self.n_bins_lgE = 1
        if sampler == "auto":
            self.sampler = f_sampler(args)
        elif sampler == 0:
            self.sampler = None  # isotropy
        else:
            self.sampler = sampler
        self.seed = seed
        self.return_frac = return_frac
        self.sigmaLnE = args.sigmaLnE
        self.logEmin = np.log(args.EminData)
        if add_iso is None:
            self.add_iso = args.f_src_min > 0 or not args.log_sample
        else:
            self.add_iso = add_iso
        self.sources = sources
        self.__args = args
        self.exclude_energy = args.exclude_energy

        if n_samples is None:
            n_samples = args.n_samples

        if deterministic is None:
            deterministic = args.deterministic

        self.deterministic = deterministic
        self.batch_size = batch_size if batch_size is not None else args.batch_size
        self.n_batchs = int(np.ceil(n_samples / self.batch_size))
        batch_size = int(np.round(n_samples / self.n_batchs))
        if self.add_iso:
            batch_size = (batch_size//2)*2  # make sure batch_size is divisible of 2

        if batch_size != self.batch_size:
            print('batch size adjusted to ', batch_size, file=stderr)
            self.batch_size = batch_size

        self.Neecr = args.Neecr
        self.coordinates = []  # x,y,z,log(E)
        self.source_weights = None
        iso_flux = np.loadtxt(args.data_dir + '/iso_flux')
        E = iso_flux[:, 0] / 1e18  # EeV
        fE = np.sum(iso_flux[:, 1:], axis=1)/E
        lnE = np.log(E)
        idx = np.where(lnE >= np.log(args.EminData)-3*self.sigmaLnE)[0]
        self.lnE_iso = lnE[idx]
        self.p_iso = fE[idx]
        self.p_iso /= np.sum(self.p_iso)

        if self.sampler is not None:  # not isotropy
            data_list = list(load_src_sample(args, suffix=suffix, sources=sources, mf=mf))
            if len(mixture)>0:
                assert len(mixture) == len(data_list), 'inconsistent mixture fractions'
                self.source_weights = np.array(mixture) / np.sum(mixture)

            # 2. Find non-zero lines, i.e., those with Z>0:

            for data in data_list:
                # Filtering invalid entries
                data = data[data[:, 5] > 0]
                if len(data) < args.Neecr:
                    logging.warning('src_sample data size is less then Neecr')
                    # this is just warning since we still can sample with replacement
                if len(data) < args.Neecr//2:
                    assert False, 'src_sample data size is less then Neecr/2'

                l_deg = data[:, 1]
                b_deg = data[:, 0]
                energy = data[:, 6]
                src_cells_file = data[:, 7].astype(np.int)
                src_cells_cur_grid = hp.ang2pix(self.__args.Nside, l_deg,
                                       b_deg, lonlat=True)
                if np.sum(src_cells_file != src_cells_cur_grid) > 0:
                    logging.warning(f'healpix grid index check failed. Map will be converted to Nside={self.__args.Nside})')
                l_deg, b_deg = hp.pix2ang(self.__args.Nside, src_cells_cur_grid, lonlat=True)

                c = SkyCoord(l=l_deg * u.degree, b=b_deg * u.degree, frame='galactic')
                xyz = np.array(c.galactic.cartesian.xyz).transpose()
                x4 = np.log(energy).reshape((-1,1))
                coordinates = np.hstack((xyz, x4))  # x,y,z,(E/EeV)^-2
                self.coordinates.append(coordinates)

                if exposure is not None:
                    # TODO: take into account exposure energy dependence for iso component
                    assert not exposure.energy_dependent, "exposure energy dependece not supported in unbinned mode"

                    points_exposure = exposure.gal_exposure(l_deg, b_deg, energy)
                    tot_exposure = np.sum(points_exposure)
                    if tot_exposure == 0:
                        raise ValueError('exposure in the direction of source is equal to 0')
                    n_non_zero_points = np.sum(points_exposure > 0)
                    if n_non_zero_points < self.Neecr:
                        logging.warning(f'number of nonzero exposure points is {n_non_zero_points}')
                    points_exposure /= tot_exposure
                    self.point_exposure.append(points_exposure)

        self.Nside = args.Nside
        self.threshold = args.threshold

    def __len__(self):
        return self.n_batchs

    def __getitem__(self, idx):
        if self.deterministic and idx == 0:
            np.random.seed(idx + self.seed)

        answers = np.zeros(self.batch_size)
        batch = []
        for i in range(self.batch_size):
            if self.sampler is None or (self.add_iso and i % 2 == 0):
                Nsrc = 0
                Niso = self.Neecr
            else:
                Nsrc, Niso = self.sampler.__next__()
            coordinates = None
            if Nsrc > 0:
                if self.source_weights is not None:  # mixture of events from different sources in one sample
                    sampled_src = np.random.choice(len(self.source_weights), Nsrc, p=self.source_weights)
                    counts = zip(*np.unique(sampled_src, return_counts=True))
                else:  # samples, containing events from single source
                    f_idx = 0
                    if len(self.coordinates) > 1:
                        f_idx = np.random.randint(0, len(self.coordinates))  # select random file
                    counts = [(f_idx, Nsrc)]
                for file_idx, n_src in counts:
                    coordinates = self.coordinates[file_idx]
                    log_energies = coordinates[:,3] + np.random.randn(len(coordinates)) * self.sigmaLnE
                    idxs = np.where(log_energies > self.logEmin)[0]

                    if len(idxs) < n_src//2 + 1:
                        assert False, 'too few points to sample from'
                    coordinates = coordinates[idxs]
                    if self.point_exposure:
                        p = self.point_exposure[file_idx][idxs]
                        p = p/np.sum(p)
                        src_sample = np.random.choice(len(coordinates), n_src, p=p, replace=True)
                    else:
                        src_sample = np.random.choice(len(coordinates), n_src, replace=True)

                    coordinates = coordinates[src_sample]
            if Niso > 0:
                if self.exposure is None:
                    # A sample of events from the isotropic background
                    # np.random.seed(iso_random_seed)
                    lon_iso = np.random.uniform(-np.pi, np.pi, Niso)
                    lat_iso = np.arccos(np.random.uniform(-1, 1, Niso)) - np.pi / 2.
                else:
                    n_iso_points = 10 * (Niso + 1)
                    lon_iso = np.random.uniform(-np.pi, np.pi, n_iso_points)
                    lat_iso = np.arccos(np.random.uniform(-1, 1, n_iso_points)) - np.pi / 2.
                    p = self.exposure.gal_exposure(lon_iso*180/np.pi, lat_iso*180/np.pi)
                    p /= np.sum(p)
                    idxs = np.random.choice(n_iso_points, Niso, p=p, replace=True)
                    lon_iso = lon_iso[idxs]
                    lat_iso = lat_iso[idxs]
                lnE = np.array([])
                while len(lnE) < Niso:
                    lnE_cur = np.random.choice(self.lnE_iso, Niso, p=self.p_iso, replace=True)
                    lnE_cur += np.random.randn(len(lnE_cur)) * self.sigmaLnE
                    lnE = np.concatenate((lnE, lnE_cur[lnE_cur >= self.logEmin]))
                lnE = lnE[:Niso]

                # Since currently we use generated healpix maps for source events we also convert iso events to
                # healpix coordinates and back to prevent classifier to learn that only source events are located
                # in the healpix cell coordinates
                # Also we emulate final angular resolution in this way
                iso_cells = hp.ang2pix(self.__args.Nside, np.rad2deg(lon_iso),
                                       np.rad2deg(lat_iso), lonlat=True)
                lon_deg, lat_deg = hp.pix2ang(self.__args.Nside, iso_cells, lonlat=True)

                c = SkyCoord(l=lon_deg * u.degree, b=lat_deg * u.degree, frame='galactic')
                xyz = np.array(c.galactic.cartesian.xyz).transpose()
                x4 = lnE.reshape((-1, 1))
                iso_coordinates = np.hstack((xyz, x4))  #
                if coordinates is None:
                    coordinates = iso_coordinates
                else:
                    coordinates = np.concatenate((coordinates, iso_coordinates), axis=0)
                E = np.exp(coordinates[:, 3])
                coordinates[:, 3] = 1000/(E*E)  # x,y,z,(E/EeV)^-2

            answers[i] = Nsrc / self.Neecr
            batch.append(coordinates)

        if not self.return_frac:
            answers = (answers > self.threshold)
        batch_features = np.array(batch)
        if self.exclude_energy:
            batch_features = batch_features[:,:,:3]

        return batch_features, answers


def main():
    init_train_cline_args('Train dinamic graph convolutional network')
    add_arg('--disable_dinamic_conv', action='store_true', help='disable dinamic convolutions (use standard graph convolutions)')
    add_arg('--use_energy_as_feature', action='store_true', help='use energy as feature only (not coordinate)')

    args = cl_args()

    if args.source_id == 'all':
        args.source_id = ','.join(sorted(source_data.keys()))

    sources = sorted(args.source_id.split(','))

    if not args.show_fig:
        matplotlib.use('Agg')  # enable figure generation without running X server session

    args.loss = get_loss(args.loss)

    train_gen = SampleGenerator(args, seed=train_seed)
    val_gen = SampleGenerator(args, deterministic=True, seed=val_seed, n_samples=args.n_validation_samples, mf=args.validation_mf)
    test_gen = [SampleGenerator(args, deterministic=True, seed=test_seed, n_samples=args.n_test_samples)]
    if len(sources) > 1:
        test_gen += [SampleGenerator(args, deterministic=True, seed=test_seed,
                                     n_samples=args.n_test_samples, sources=[s]) for s in sources]

    test_b_gen = [SampleGenerator(args, deterministic=True, seed=test_seed, n_samples=args.n_test_samples,
                                      mf=args.test_mf)]
    if len(sources) > 1:
        test_b_gen += [SampleGenerator(args, deterministic=True, seed=test_seed, n_samples=args.n_test_samples,
                                       sources=[s], mf=args.test_mf) for s in sources]


    def train_model(model, save_name, epochs=400, verbose=1, n_early_stop_epochs=30, batch_size=1024):
        for i in range(args.min_version, 100000):
            save_path = save_name + "_v" + str(i) + '.h5'
            if not path.isfile(save_path):
                with open(save_path, mode='x') as f:
                    pass
                break

        weights_file = '/tmp/' + path.basename(save_path) + '_best_weights.h5'
        frac_log_file = save_path[:-3] + '_det_frac.txt'
        frac_log = open(frac_log_file, mode='wt', buffering=1)
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
                    keras.callbacks.EarlyStopping(monitor=args.monitor, patience=n_early_stop_epochs, verbose=1, min_delta=args.min_delta)
                    # early stop
                )

        validation_data = None if args.monitor.startswith('frac') else val_gen
        t = time.time()
        history = model.fit_generator(train_gen, epochs=epochs, verbose=verbose,
                            validation_data=validation_data, callbacks=callbacks)
        t = time.time() - t

        if n_early_stop_epochs > 0 and path.isfile(weights_file):
            model.load_weights(weights_file)  # load best weights
            # remove(weights_file)

        print('Training took %.0f sec' % t)
        model.save(save_path)
        print('Model saved in', save_path)

        if n_early_stop_epochs > 0 and path.isfile(weights_file):
            remove(weights_file)

        if args.plot_learning_curves:
            plot_learning_curves(history, save_file=save_path[:-3] + '_train.png', show_fig=args.show_fig)

        if args.evaluate_test_loss:
            score = model.evaluate_generator(test_gen[0], verbose=0)
        else:
            score = []

        with open(save_path + '.args', mode='w') as args_out:
            from sys import argv
            print(*argv, file=args_out)
            print(file=args_out)
            for key, val in args.__dict__.items():
                print(key, '=', val, file=args_out)

        with open(save_path + '.score', mode='w') as out:
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

    train_model(model, save_name, batch_size=args.batch_size, epochs=args.n_epochs,
                n_early_stop_epochs=args.n_early_stop)

if __name__ == '__main__':
    main()
