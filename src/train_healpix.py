import logging

import numpy as np
import argparse
from tensorflow import keras
import time
import matplotlib
from os import path, remove
import healpy as hp
from sys import stderr, stdout
from cnn_healpix import create_model
from beta import calc_detectable_frac
from train import *

def SampleGenerator(args, **kwargs):
    from exposure import create_exposure
    exposure = create_exposure(args)
    if args.exclude_energy:
        return SampleGeneratorSingleBin(args, exposure=exposure, **kwargs)
    else:
        return SampleGeneratorWithEbins(args, exposure=exposure, **kwargs)


class SampleGeneratorSingleBin(keras.utils.Sequence):
    def __init__(self, args, deterministic=None, seed=0, n_samples=None, return_frac=False, suffix='*', sources=None,
                 mixture=[], add_iso=None, sampler="auto", batch_size=None, mf=None, exposure=None):
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
        if add_iso is None:
            self.add_iso = args.f_src_min > 0 or not args.log_sample
        else:
            self.add_iso = add_iso
        self.sources = sources
        self.__args = args

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
        self.Ncells = hp.nside2npix(args.Nside)
        self.healpix_src_cells = []
        self.source_weights = None

        if self.sampler is not None:  # not isotropy
            data_list = list(load_src_sample(args, suffix=suffix, sources=sources, mf=mf))
            if len(mixture)>0:
                assert len(mixture) == len(data_list), 'inconsistent mixture fractions'
                self.source_weights = np.array(mixture) / np.sum(mixture)

            # 2. Find non-zero lines, i.e., those with Z>0:

            for data in data_list:
                # Filtering invalid entries
                data = data[data[:, 5] > 0]
                assert len(data) >= args.Neecr

                # saving numbers of cells on the healpix grid occupied by the nuclei
                self.healpix_src_cells.append(data[:, 7].astype(int))
                if exposure is not None:
                    # TODO: take into account exposure energy dependence for iso component
                    assert not exposure.energy_dependent, "exposure energy dependece not supported in unbinned mode"
                    l_deg = data[:, 3]
                    b_deg = data[:, 2]
                    energy = data[:, 6]
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
        if self.deterministic:
            np.random.seed(idx + self.seed)

        healpix_map = np.zeros((self.batch_size, self.Ncells), dtype=float)
        answers = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            if self.sampler is None or (self.add_iso and i % 2 == 0):
                Nsrc = 0
                Niso = self.Neecr
            else:
                Nsrc, Niso = self.sampler.__next__()

            if Nsrc > 0:
                if self.source_weights is not None:  # mixture of events from different sources in one sample
                    sampled_src = np.random.choice(len(self.source_weights), Nsrc, p=self.source_weights)
                    counts = zip(*np.unique(sampled_src, return_counts=True))
                else:  # samples, containing events from single source
                    f_idx = 0
                    if len(self.healpix_src_cells) > 1:
                        f_idx = np.random.randint(0, len(self.healpix_src_cells))  # select random file
                    counts = [(f_idx, Nsrc)]
                for file_idx, n_src in counts:
                    healpix_src_cells = self.healpix_src_cells[file_idx]
                    nonz_number = len(healpix_src_cells)

                    # Create a random sample of from-source events:
                    if self.point_exposure:
                        src_sample = np.random.choice(nonz_number, n_src, p=self.point_exposure[file_idx], replace=True)
                    else:
                        src_sample = np.random.choice(nonz_number, n_src, replace=False)

                    # Cells of the healpix grid "occupied" by the from-source sample
                    src_cells = healpix_src_cells[src_sample]

                    # This is a simple way to define the map. It assumes
                    # only one EECR gets in a cell. Works OK for Nside=512.
                    # healpix_map[src_cells] = 1

                    # "cells" is an intermediate variable to simplify the code.
                    # It consists of two arrays: [0] is a list of unique cells,
                    # [1] is their multiplicity.
                    cells = np.unique(src_cells, return_counts=True)
                    healpix_map[i, cells[0]] += cells[1]

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

                # Cells "occupied" by the isotropic sample in the healpix grid
                iso_cells = hp.ang2pix(self.Nside, np.rad2deg(lon_iso),
                                       np.rad2deg(lat_iso), lonlat=True)

                # healpix_map[iso_cells] = 1
                # Similar to the above
                cells = np.unique(iso_cells, return_counts=True)
                healpix_map[i, cells[0]] += cells[1]

            answers[i] = Nsrc / self.Neecr

        if not self.return_frac:
            answers = (answers > self.threshold)

        return healpix_map, answers


class SampleGeneratorWithEbins(keras.utils.Sequence):
    def __init__(self, args, deterministic=None, seed=0, n_samples=None, return_frac=False, suffix='*', sources=None,
                 mixture=[], add_iso=None, sampler="auto", batch_size=None, mf=None, exposure=None):
        self.point_exposure = []
        self.exposure = exposure
        assert exposure is None, "Not implemented yet"
        min_lg_E = np.log10(args.EminData)
        sigmaLgE = args.sigmaLnE / np.log(10.)
        eminSigmaDif = args.EminSigmaDif

        if min_lg_E - np.log10(args.Emin) < eminSigmaDif * sigmaLgE:
            validEmin = 10**(min_lg_E - eminSigmaDif * sigmaLgE)
            validEminBin = 10**(np.log10(args.Emin) + eminSigmaDif * sigmaLgE)
            message = f'use Emin < {validEmin} or EminData >{validEminBin} or choose smaller sigmaLnE (difference must be larger then {eminSigmaDif} sigma)'
            raise ValueError(message)

        lgEbin = args.lgEbin
        self.n_bins_lgE = int((np.log10(args.Emax)-min_lg_E)/lgEbin + 0.5)
        from scipy.stats import norm
        n_weight_calc_bins = 2 * self.n_bins_lgE  # can be any sufficiently large number
        bin_diff_weights = np.zeros(n_weight_calc_bins)
        for nb in range(n_weight_calc_bins):
            x1 = (nb - 0.5) * lgEbin
            x2 = (nb + 0.5) * lgEbin
            bin_diff_weights[nb] = norm.cdf(x2, scale=sigmaLgE) - norm.cdf(x1, scale=sigmaLgE)
        bin_diff_weights = np.hstack((np.flip(bin_diff_weights), bin_diff_weights[1:]))

        iso_flux = np.loadtxt(args.data_dir + '/iso_flux')
        fE2 = np.sum(iso_flux[:, 1:], axis=1)
        E = iso_flux[:, 0] / 1e18  # EeV
        idx = np.where(E >= args.Emin)[0]
        E = E[idx]
        fE = fE2[idx] / E

        binE = ((np.log10(E) - min_lg_E) / lgEbin).astype(np.int)
        #  tp = np.arange(0, np.size(data[:, 0]))
        nonz = np.where((binE >= -self.n_bins_lgE) & (binE < self.n_bins_lgE))[0]  # tp[data[:, 5] > 0]
        binE = binE[nonz]
        fE = fE[nonz]

        weights_idx = n_weight_calc_bins - binE - 1
        weights = [bin_diff_weights[idx:idx + self.n_bins_lgE] for idx in weights_idx]
        probabilities = np.vstack(weights)  # increase n_weight_calc_bins if weights entries have different sizes
        probabilities *= fE.reshape((-1, 1))
        probabilities /= np.sum(probabilities)  # normalize to unit total
        self.energy_probabilities = np.sum(probabilities, axis=0)

        if sampler == "auto":
            self.sampler = f_sampler(args)
        elif sampler == 0:
            self.sampler = None  # isotropy
        else:
            self.sampler = sampler
        self.seed = seed
        self.return_frac = return_frac
        if add_iso is None:
            self.add_iso = args.f_src_min > 0 or not args.log_sample
        else:
            self.add_iso = add_iso
        self.sources = sources
        self.__args = args

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
        self.Ncells = hp.nside2npix(args.Nside)
        self.healpix_src_cells = []
        self.probabilities = []

        self.source_weights = None
        if self.sampler is not None:  # not isotropy
            data_list = list(load_src_sample(args, suffix=suffix, sources=sources, mf=mf))
            if len(mixture)>0:
                assert len(mixture) == len(data_list), 'inconsistent mixture fractions'
                self.source_weights = np.array(mixture) / np.sum(mixture)

            # 2. Find non-zero lines, i.e., those with Z>0:

            for data in data_list:
                assert len(data) > 0, "invalid input map"
                data = data[data[:, 5] > 0]
                binE = ((np.log10(data[:, 6])-min_lg_E)/lgEbin).astype(np.int)
                nonz = np.where((binE >= -self.n_bins_lgE) & (binE < self.n_bins_lgE))[0]  # tp[data[:, 5] > 0]
                assert len(nonz) >= args.Neecr
                data = data[nonz]
                binE = binE[nonz]
                healpix_src_cells = data[:, 7].astype(int)
                self.healpix_src_cells.append(healpix_src_cells)
                weights_idx = n_weight_calc_bins-binE-1
                weights = [bin_diff_weights[idx:idx + self.n_bins_lgE] for idx in weights_idx]
                probabilities = np.vstack(weights)  # increase n_weight_calc_bins if weights entries have different sizes
                probabilities /= np.sum(probabilities)  # normalize to unit total
                self.probabilities.append(probabilities.ravel())

        self.Nside = args.Nside
        self.threshold = args.threshold

    def __len__(self):
        return self.n_batchs

    def __getitem__(self, idx):
        if self.deterministic:
            np.random.seed(idx + self.seed)

        healpix_map = np.zeros((self.batch_size, self.Ncells, self.n_bins_lgE), dtype=float)
        answers = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            if self.sampler is None or (self.add_iso and i % 2 == 0):
                Nsrc = 0
                Niso = self.Neecr
            else:
                Nsrc, Niso = self.sampler.__next__()

            if Nsrc > 0:
                if self.source_weights is not None:  # mixture of events from different sources in one sample
                    sampled_src = np.random.choice(len(self.source_weights), Nsrc, p=self.source_weights)
                    counts = zip(*np.unique(sampled_src, return_counts=True))
                else:  # samples, containing events from single source
                    f_idx = 0
                    n_files = len(self.healpix_src_cells)
                    if n_files > 1:
                        f_idx = np.random.randint(0, n_files)  # select random file
                    counts = [(f_idx, Nsrc)]
                for file_idx, n_src in counts:
                    p = self.probabilities[file_idx]
                    idxs = np.random.choice(len(p), n_src, replace=True, p=p)
                    healpix_cell_idxs = idxs // self.n_bins_lgE
                    energy_bin_idxs = idxs % self.n_bins_lgE

                    #nonz_number = self.nonz_number[file_idx]
                    healpix_src_cells = self.healpix_src_cells[file_idx][healpix_cell_idxs]

                    for e_idx, cell_idx in zip(energy_bin_idxs, healpix_src_cells):
                        healpix_map[i, cell_idx, e_idx] += 1

            if Niso > 0:
                lon_iso = np.random.uniform(-np.pi, np.pi, Niso)
                lat_iso = np.arccos(np.random.uniform(-1, 1, Niso)) - np.pi / 2.
                energy_bin_idxs = np.random.choice(len(self.energy_probabilities), Niso, replace=True, p=self.energy_probabilities)
                # Cells "occupied" by the isotropic sample in the healpix grid
                iso_cells = hp.ang2pix(self.Nside, np.rad2deg(lon_iso),
                                       np.rad2deg(lat_iso), lonlat=True)

                for e_idx, cell_idx in zip(energy_bin_idxs, iso_cells):
                    healpix_map[i, cell_idx, e_idx] += 1

            answers[i] = Nsrc / self.Neecr

        if not self.return_frac:
            answers = (answers > self.threshold)

        return healpix_map, answers


def main():
    init_train_cline_args('Train healpix conv network')

    add_arg('--layers', type=str, help='comma-separated list of inner layer sizes', default='16,16')
    add_arg('--nside_min', type=int, help='minimal Nside for convolution', default=1)
    add_arg('--n_filters', type=int, help='number of filters for convolution', default=76)
    add_arg('--Emax', type=int, help='maximal binning energy in EeV', default=300)
    add_arg('--EminSigmaDif', type=float, help='minimal difference in between Emin and EminData in terms of sigma used for param validation', default=3)
    add_arg('--lgEbin', type=float, help='Log10 energy bin', default=0.05)

    args = cl_args()

    if args.source_id == 'all':
        args.source_id = ','.join(sorted(source_data.keys()))

    sources = sorted(args.source_id.split(','))

    if not args.show_fig:
        matplotlib.use('Agg')  # enable figure generation without running X server session

    inner_layers = [int(l) for l in args.layers.split(',') if len(l) > 0]

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
            remove(weights_file)

        print('Training took %.0f sec' % t)
        model.save(save_path)
        print('Model saved in', save_path)

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



    model = create_model(train_gen.Ncells, n_energy_bins=train_gen.n_bins_lgE, nside_min=args.nside_min, n_filters=args.n_filters,
                         inner_layer_sizes=inner_layers, pretrained=args.pretrained)

    if args.pretrained and len(args.output_prefix) == 0:
        save_name = args.pretrained[:-3]
    else:
        n_side_min = min(args.Nside, args.nside_min)
        if args.exposure == 'uniform':
            prefix = ''
        else:
            prefix = args.exposure + '_'
        prefix += '_'.join(sorted(args.source_id.split(',')))
        save_name = args.output_prefix + '{}_N{}_B{}_Ns{}-{}_F{}'.format(prefix, args.Neecr, args.mf, args.Nside,
                                                                     n_side_min, args.n_filters)
        if len(inner_layers) > 0:
            save_name += ("_L" + '_'.join([str(i) for i in inner_layers]))
        if args.threshold > 0:
            save_name += '_th' + str(args.threshold)
        if not args.exclude_energy:
            save_name += '_sig{:.0f}_b{:.2f}'.format(100*args.sigmaLnE, args.lgEbin)


    train_model(model, save_name, batch_size=args.batch_size, epochs=args.n_epochs,
                n_early_stop_epochs=args.n_early_stop)

if __name__ == '__main__':
    main()
