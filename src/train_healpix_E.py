import numpy as np
import argparse
import losses
from tensorflow import keras #  import   keras
import time
import matplotlib
from os import path, remove
import healpy as hp
from sys import stderr, stdout
from cnn_healpix import create_model

train_seed = 0
val_seed = 2 ** 20
test_seed = 2 ** 26

def get_loss(loss):
    # try to find loss by name in losses module
    # if not found use string as is
    try:
        loss = getattr(losses, loss)
    except AttributeError:
        pass
    return loss

source_data = {
    # Name : [source_lon, source_lat, D_src]
    'M82': [141.4095,40.5670,'3.5'],
    'CenA': [309.5159,19.4173,'3.5'],
    'NGC253': [97.3638,-87.9645,'3.5'],
    # 'NGC6946': [95.71873,11.6729,'6.0'],
    'M87': [283.7777,74.4912, '18.5'],
    'FornaxA': [240.1627,-56.6898,'20.0']
}

def get_source_data(source_id):
    if source_id in source_data:
        return tuple(source_data[source_id])
    else:
        raise ValueError('Unknown source!')


def load_src_sample(args, suffix='', sources=None):
    """
    Load data from src_sample files
    :param args: command line params (used if sources are given by name or not given at all)
    :param suffix: may be used to select a ranage of files (e.g. suffix='*') or specific realizations
    :param sources: optional explicit list of sources or files
    :return:
    """
    import lzma
    import glob

    if sources is None:
        sources = args.source_id.split(',')

    for source_id in sources:
        if 'src_sample_' in source_id:
            infiles = source_id  # looks like path
        else:
            _, _, D_src = get_source_data(source_id)  # looks like source name
            infiles = ('src_sample_' + source_id + '_D' + D_src
                      + '_Emin' + str(args.Emin)
                      + '_N' + str(args.Nini)
                      + '_R' + str(args.source_vicinity_radius)
                      + '_Nside' + str(args.Nside) + suffix
                      + '.txt.xz')
            infiles = args.data_dir + '/' + args.mf + '/sources/' + infiles
        files = list(glob.glob(infiles))
        if len(files) == 0:
            raise ValueError(infiles + ' file(s) not found!')
        for infile in files:
            with lzma.open(infile, 'rt') as f:
                yield np.genfromtxt(f, dtype=float)


def f_sampler(args, n_samples=-1, exclude_iso=False):  # if < 0, sample forever
    Neecr = args.Neecr
    Fsrc = None
    if 0 <= args.f_src <= 1.:
        Fsrc = args.f_src
    else:
        assert 0 <= args.f_src_min < 1
        assert 0 < args.f_src_max <= 1
        assert args.f_src_min < args.f_src_max

        N_src_min = np.round(args.f_src_min * Neecr)
        N_src_max = np.round(args.f_src_max * Neecr)
        if exclude_iso:
            N_src_min = max(1, N_src_min)
            N_src_max = max(1, N_src_max)

        if N_src_min == N_src_max:
            Fsrc = N_src_min / Neecr
        elif args.log_sample:
            if args.f_src_min == 0 and not exclude_iso:
                # make sure roughly equal amount of isotropic and mixture samples are generated
                logNmin = np.log(1. / N_src_max)
            else:
                logNmin = np.log((N_src_min-0.49))
            logNmax = np.log((N_src_max + 0.49))

    n = 0
    if Fsrc is not None:
        Nsrc = int(np.round(Neecr*Fsrc))

    while n != n_samples:
        if Fsrc is None:
            if args.log_sample:
                Nsrc = int(np.round(np.exp(logNmin + (logNmax-logNmin)*np.random.rand())))
                assert N_src_min <= Nsrc <= N_src_max
            else:
                Nsrc = np.random.randint(N_src_min, N_src_max+1)

        Niso = Neecr - Nsrc
        yield (Nsrc, Niso)
        n += 1

class SampleGenerator(keras.utils.Sequence):
    def __init__(self, args, deterministic=None, seed=0, n_samples=None, return_frac=False, suffix='*', sources=None,
                 mixture=[], add_iso=None, sampler="auto", batch_size=None):
        min_lg_E = np.log10(args.EminBin)
        sigmaLgE = args.sigmaLnE / np.log(10.)
        eminSigmaDif = 3

        if min_lg_E - np.log10(args.Emin) < eminSigmaDif * sigmaLgE:
            validEmin = 10**(min_lg_E - eminSigmaDif * sigmaLgE)
            validEminBin = 10**(np.log10(args.Emin) + eminSigmaDif * sigmaLgE)
            message = 'use Emin < {} or EminBin >{} or choose smaller sigmaLnE (difference must be larger then {} sigma)'.format(validEmin, validEminBin, eminSigmaDif)
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
        self.energy_probabilities = []
        self.nonz_number = []
        self.source_part = None
        if self.sampler is not None:  # not isotropy
            data_list = list(load_src_sample(args, suffix=suffix, sources=sources))
            if len(mixture)>0:
                assert len(mixture) == len(data_list), 'inconsistent mixture fractions'
                self.source_part = np.array(mixture)/np.sum(mixture)

            # 2. Find non-zero lines, i.e., those with Z>0:

            for data in data_list:
                nonz = np.where(data[:, 5] > 0)[0]
                data = data[nonz]
                binE = ((np.log10(data[:, 6])-min_lg_E)/lgEbin).astype(np.int)
                #  tp = np.arange(0, np.size(data[:, 0]))
                nonz = np.where((binE >= -self.n_bins_lgE) & (binE < self.n_bins_lgE))[0]  # tp[data[:, 5] > 0]
                assert len(nonz) >= args.Neecr

                # These are numbers of cells on the healpix grid occupied by the
                # nuclei in the infile

                healpix_src_cells = data[nonz, 7].astype(int)
                self.healpix_src_cells.append(healpix_src_cells)
                binE = binE[nonz]
                weights_idx = n_weight_calc_bins-binE-1
                weights = [bin_diff_weights[idx:idx + self.n_bins_lgE] for idx in weights_idx]
                probabilities = np.vstack(weights)  # increase n_weight_calc_bins if weights entries have different sizes
                probabilities /= np.sum(probabilities)  # normalize to unit total
                self.energy_probabilities.append(np.sum(probabilities, axis=0))
                self.probabilities.append(probabilities.ravel())
                self.nonz_number.append(len(nonz))

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
                if self.source_part is not None:  # mixture of events from different sources in one sample
                    sampled_src = np.random.choice(len(self.source_part ), Nsrc, p=self.source_part)
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
                # ensure same average energy spectrum is used as in anisotropic case
                if self.source_part is not None:  # mixture of events from different sources in one sample
                    sampled_src = np.random.choice(len(self.source_part ), Niso, p=self.source_part)
                    counts = zip(*np.unique(sampled_src, return_counts=True))
                else:  # samples, containing events from single source
                    f_idx = 0
                    n_files = len(self.healpix_src_cells)
                    if n_files > 1:
                        f_idx = np.random.randint(0, n_files)  # select random file
                    counts = [(f_idx, Niso)]

                for file_idx, n_src in counts:
                    p = self.energy_probabilities[file_idx]
                    energy_bin_idxs = np.random.choice(len(p), n_src, replace=True, p=p)

                    # A sample of events from the isotropic background
                    # np.random.seed(iso_random_seed)
                    lon_iso = np.random.uniform(-np.pi, np.pi, n_src)
                    lat_iso = np.arccos(np.random.uniform(-1, 1, n_src)) - np.pi / 2.

                    # Cells "occupied" by the isotropic sample in the healpix grid
                    iso_cells = hp.ang2pix(self.Nside, np.rad2deg(lon_iso),
                                           np.rad2deg(lat_iso), lonlat=True)

                    for e_idx, cell_idx in zip(energy_bin_idxs, iso_cells):
                        healpix_map[i, cell_idx, e_idx] += 1

            answers[i] = Nsrc / self.Neecr

        if not self.return_frac:
            answers = (answers > self.threshold)

        return healpix_map, answers

def plot_learning_curves(history, save_file=None, show_fig=False):
    import matplotlib.pyplot as plt
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
        if show_fig:
            plt.show()
    except Exception:
        pass


def calc_beta(gen, model, _alpha, gen2=None):
    """
    :param gen: sample generator
    :param model: NN model
    :param alpha: maximal type I error
    :param gen2: if gen2 is not None gen output is used for 0-hypothesis and gen2 for alternative
    otherwise frac > 0 condition is used
    :return: (frac, alpha) minimal fraction of source events in alternative (gen2) hypothesis and precise alpha or (1., 1.) if detection is impossible
    """
    data = [gen] if gen2 is None else [gen, gen2]
    src = xi = frac = None
    for i, g in enumerate(data):
        save = g.return_frac
        g.return_frac = True
        for batch in range(len(g)):
            maps, batch_frac = g.__getitem__(batch)
            batch_xi = model.predict(maps).flatten()
            if gen2 is None:
                batch_src = batch_frac > 0
            else:
                batch_src = np.full(len(batch_frac), i == 0)
            if src is None:
                src = batch_src
                xi = batch_xi
                frac = batch_frac
            else:
                src = np.concatenate((src, batch_src))
                xi = np.concatenate((xi, batch_xi))
                frac = np.concatenate((frac, batch_frac))
        g.return_frac = save

    h0 = np.logical_not(src)  # is 0-hypothesis
    h0_xi = xi[h0]
    xi = xi[src]

    if np.mean(h0_xi) > np.mean(xi):  # below we assume <h0_xi>  <=  <xi>
        xi *= -1.
        h0_xi *= -1.

    alpha_thr = np.quantile(h0_xi, 1. - _alpha)

    # sort by xi
    xi = np.sort(xi)

    thr_idx = np.where(xi >= alpha_thr)[0][0]
    beta = thr_idx/len(xi)

    return beta


def calc_detectable_frac(gen, model, args, gen2=None, swap_h0_and_h1=False):
    """
    :param gen: sample generator
    :param model: NN model
    :param args: parameters object (should contain alpha and beta attributes)
    :param gen2: if gen2 is not None gen output is used for 0-hypothesis and gen2 for alternative
    otherwise frac > 0 condition is used
    :return: (frac, alpha) minimal fraction of source events in alternative (gen2) hypothesis and precise alpha or (1., 1.) if detection is impossible
    """
    if swap_h0_and_h1:
        _alpha = args.beta
        _beta = args.alpha
    else:
        _alpha = args.alpha
        _beta = args.beta
    data = [gen] if gen2 is None else [gen, gen2]
    src = xi = frac = None
    for i, g in enumerate(data):
        save = g.return_frac
        g.return_frac = True
        for batch in range(len(g)):
            maps, batch_frac = g.__getitem__(batch)
            batch_xi = model.predict(maps).flatten()
            if gen2 is None:
                batch_src = batch_frac > 0
            else:
                batch_src = np.full(len(batch_frac), i == 0)
            if src is None:
                src = batch_src
                xi = batch_xi
                frac = batch_frac
            else:
                src = np.concatenate((src, batch_src))
                xi = np.concatenate((xi, batch_xi))
                frac = np.concatenate((frac, batch_frac))
        g.return_frac = save

    h0 = np.logical_not(src)  # is 0-hypothesis
    h0_xi = xi[h0]
    fractions = frac[src]
    xi = xi[src]

    if np.mean(h0_xi) > np.mean(xi):  # below we assume <h0_xi>  <=  <xi>
        xi *= -1.
        h0_xi *= -1.

    alpha_thr = np.quantile(h0_xi, 1. - _alpha)

    # sort by xi
    idx = np.argsort(xi)
    fractions = fractions[idx]
    xi = xi[idx]

    thr_idx = np.where(xi >= alpha_thr)[0][0]

    fracs = sorted(list(set(fractions)))

    def beta(i_f):
        idx = np.where(fractions >= fracs[i_f])[0]   # TODO: fix bug: >= is wrong
        idx_left = np.where(idx < thr_idx)[0]
        return len(idx_left) / len(idx)

    def alpha(i_f):
        thr = np.quantile(xi[fractions >= fracs[i_f]], _beta)
        idx_right = np.where(h0_xi > thr)[0]
        return len(idx_right) / len(h0_xi)

    l = 0
    r = len(fracs) - 1

    beta_l = beta(l)
    beta_r = beta(r)

    if beta_r > _beta:
        print('solution not found', file=stderr)
        return 1., 1.

    if beta_r == _beta:
        l = r

    if beta_l <= _beta:
        print('all mixed samples satisfy criterion', file=stderr)
        r = l

    i = (l + r) // 2

    while r > l + 2:
        b = beta(i)
        if b > _beta:
            l = i
        elif b < _beta:
            r = i
        else:
            break
        i = (l + r) // 2

    if beta(i) > _beta:
        i += 1

    if swap_h0_and_h1:
        type_I_error_P = beta(i)
    else:
        type_I_error_P = alpha(i)

    return fracs[i], type_I_error_P

def main():
    cline_parser = argparse.ArgumentParser(description='Train network',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    def add_arg(*pargs, **kwargs):
        cline_parser.add_argument(*pargs, **kwargs)

    lgEbin = 0.05

    add_arg('--f_src', type=float, help='fraction of "from-source" EECRs [0,1] or -1 for random', default=-1)
    add_arg('--Neecr', type=int, help='Total number of EECRs in each sample', default=500)
    add_arg('--Emin', type=int, help='Emin in EeV for which the input sample was generated', default=56)
    add_arg('--EminBinShift', type=int, help='Shift between minimal bin energy and Emin', default=2)
    # add_arg('--Nmixed_samples', type=int, help='Number of mixed samples (i.e., the sample size)', default=1000)
    add_arg('--source_id', type=str, help='source (CenA, NGC253, M82, M87 or FornaxA) or comma separated list of sources or "all"',
            default='CenA')
    add_arg('--data_dir', type=str, help='data root directory (should contain jf/sources/ or pt/sources/)',
            default='data')
    add_arg('--mf', type=str, help='Magnetic field model (jf or pt)', default='jf')
    add_arg('--compare_mf', type=str, help='Magnetic field model to compare with (jf or pt)', default='')
    add_arg('--Nside', type=int, help='healpix grid Nside parameter', default=32)
    add_arg('--Nini', type=int, help='Size of the initial sample of from-source events', default=10000)
    add_arg('--log_sample', action='store_true', help="sample f_src uniformly in log scale")
    add_arg('--f_src_max', type=float, help='maximal fraction of "from-source" EECRs [0,1]', default=1)
    add_arg('--f_src_min', type=float, help='minimal fraction of "from-source" EECRs [0,1]', default=0)
    add_arg('--layers', type=str, help='comma-separated list of inner layer sizes', default='')
    add_arg('--output_prefix', type=str, help='output model file path prefix', default='')
    add_arg('--batch_size', type=int, help='size of training batch', default=100)
    add_arg('--n_epochs', type=int, help='number of training epochs', default=1000)
    add_arg('--show_fig', action='store_true', help="Show learning curves")
    add_arg('--n_early_stop', type=int, help='number of epochs to monitor for early stop', default=10)
    add_arg('--pretrained', type=str, help='pretrained network', default='')
    add_arg('--loss', type=str, help='NN loss', default='binary_crossentropy')
    add_arg('--monitor', type=str, help='NN metrics: used for early stop val_loss/frac/frac_compare', default='frac')
    add_arg('--n_samples', type=int, help='number of samples', default=50000)
    add_arg('--nside_min', type=int, help='minimal Nside for convolution', default=1)
    add_arg('--n_filters', type=int, help='number of filters for convolution', default=32)
    add_arg('--source_vicinity_radius', type=str, help='source vicinity radius', default='1')
    add_arg('--threshold', type=float,
            help='source fraction threshold for binary classification', default=0.0)
    add_arg('--deterministic', action='store_true', help="use deterministic batches for training (default is random)")
    add_arg('--alpha', type=float, help='type 1 maximal error', default=0.01)
    add_arg('--beta', type=float, help='type 2 maximal error', default=0.05)
    add_arg('--min_version', type=int, help='minimal version number for output naming', default=0)
    add_arg('--sigmaLnE', type=float, help='deltaE/E energy resolution', default=0.2)
    add_arg('--lgEbin', type=float, help='Log10 energy bin', default=lgEbin)
    add_arg('--Emax', type=int, help='maximal binning energy in EeV', default=300)
    add_arg('--EminBin', type=float, help='minimal binning energy in EeV', default=56)

    args = cline_parser.parse_args()

    if args.source_id == 'all':
        args.source_id = ','.join(sorted(source_data.keys()))

    sources = sorted(args.source_id.split(','))

    if not args.show_fig:
        matplotlib.use('Agg')  # enable figure generation without running X server session

    inner_layers = [int(l) for l in args.layers.split(',') if len(l) > 0]

    args.loss = get_loss(args.loss)

    train_gen = SampleGenerator(args, seed=train_seed)
    val_gen = SampleGenerator(args, deterministic=True, seed=val_seed, n_samples=max(1000, args.n_samples//10))
    test_gen = [SampleGenerator(args, deterministic=True, seed=test_seed, n_samples=max(10000, args.n_samples))]
    if len(sources) > 1:
        test_gen += [SampleGenerator(args, deterministic=True, seed=2 ** 26,
                                     n_samples=max(10000, args.n_samples), sources=[s]) for s in sources]

    frac_gen = test_gen[0]
    frac_mf = args.mf

    test_b_gen = None
    if len(args.compare_mf)==0:
        args.compare_mf = 'pt' if args.mf == 'jf' else 'jf'

    mf = args.mf
    args.mf = args.compare_mf
    try:
        test_b_gen = [SampleGenerator(args, deterministic=True, seed=2**26, n_samples=max(10000, args.n_samples))]
        if len(sources) > 1:
            test_b_gen += [SampleGenerator(args, deterministic=True, seed=2**26,
                                         n_samples=max(10000, args.n_samples), sources=[s]) for s in sources]

        if args.monitor == 'frac_compare':
            frac_gen = test_b_gen[0]
            frac_mf = args.compare_mf
    except ValueError as ver:
        print(args.compare_mf, 'field test will be skipped:', str(ver))
        if args.monitor == 'frac_compare':
            args.monitor == 'frac'

    args.mf = mf

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
            f_a = calc_detectable_frac(frac_gen, model, args)
            frac_and_alphas.append(f_a)
            print(epoch, f_a[0], f_a[1], file=frac_log)
            print('Detectable fraction:', f_a[0], '\talpha =', f_a[1])
            if args.monitor.startswith('frac'):
                f_a_sorted = sorted(frac_and_alphas)

                if len(frac_and_alphas) == 1 or f_a == f_a_sorted[0]:
                    model.save_weights(weights_file, overwrite=True)
                elif n_early_stop_epochs > 0 and len(frac_and_alphas) - frac_and_alphas.index(f_a_sorted[0]) > n_early_stop_epochs:
                    model.stop_training = True
                    print('Early stop on epoch', epoch)


        callbacks = [keras.callbacks.LambdaCallback(on_epoch_end=frac_logging_callback)]
        if not args.monitor.startswith('frac'):
            callbacks.append(
                keras.callbacks.ModelCheckpoint(weights_file, save_best_only=True, monitor=args.monitor,
                                                save_weights_only=True)  # save best model
            )
            if n_early_stop_epochs > 0:
                callbacks.append(
                    keras.callbacks.EarlyStopping(monitor=args.monitor, patience=n_early_stop_epochs, verbose=1)
                    # early stop
                )

        t = time.time()

        frac_gen.seed = val_seed  # make sure different samples are used in validation and test phase
        history = model.fit_generator(train_gen, epochs=epochs, verbose=verbose,
                            validation_data=val_gen, callbacks=callbacks)
        t = time.time() - t
        frac_gen.seed = test_seed  # make sure different samples are used in validation and test phase
        if n_early_stop_epochs > 0 and path.isfile(weights_file):
            model.load_weights(weights_file)  # load best weights
            remove(weights_file)

        print('Training took %.0f sec' % t)
        model.save(save_path)
        print('Model saved in', save_path)

        plot_learning_curves(history, save_file=save_path[:-3] + '_train.png', show_fig=args.show_fig)
        score = model.evaluate_generator(test_gen[0], verbose=0)

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

            frac_and_alphas.sort()
            f, a = frac_and_alphas[0]
            for file in [out, stdout]:

                print('best_val_frac_' + frac_mf, f, file=file)
                print('best_val_alpha_' + frac_mf, a, file=file)

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
                    # print(args.compare_mf, 'field..')
                    test_frac, test_alpha = calc_detectable_frac(b_gen[0], model, args)
                    for file in [out, stdout]:
                        print('frac_' + src_name + '_' + args.compare_mf, test_frac, file=file)
                        print('alpha_' + src_name + '_' + args.compare_mf, test_alpha, file=file)



    model = create_model(train_gen.Ncells, n_energy_bins=train_gen.n_bins_lgE, nside_min=args.nside_min, n_filters=args.n_filters,
                         inner_layer_sizes=inner_layers, pretrained=args.pretrained)

    if args.pretrained and len(args.output_prefix) == 0:
        save_name = args.pretrained[:-3]
    else:
        n_side_min = min(args.Nside, args.nside_min)
        prefix = '_'.join(sorted(args.source_id.split(',')))
        save_name = args.output_prefix + '{}_N{}_B{}_Ns{}-{}_F{}'.format(prefix, args.Neecr, args.mf, args.Nside,
                                                                     n_side_min, args.n_filters)
        if len(inner_layers) > 0:
            save_name += ("_L" + '_'.join([str(i) for i in inner_layers]))
        if args.threshold > 0:
            save_name += '_th' + str(args.threshold)

    train_model(model, save_name, batch_size=args.batch_size, epochs=args.n_epochs,
                n_early_stop_epochs=args.n_early_stop)

if __name__ == '__main__':
    main()

