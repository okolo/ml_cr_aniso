import numpy as np

def calc_beta_eta(gen, model, alpha, gen2=None, beta_threshold=None):
    """
    :param gen: sample generator
    :param model: NN model
    :param args: parameters object (should contain alpha and beta attributes)
    :param gen2: if gen2 is not None gen output is used for 0-hypothesis and gen2 for alternative
    otherwise frac > 0 condition is used
    :param beta_threshold: threshold beta value for which minimal fraction is calculated
    :return: tuple of arrays (frac, beta) beta as function of fraction of source events in alternative (gen2) hypothesis
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
    fractions = frac[src]
    xi = xi[src]

    if np.mean(h0_xi) > np.mean(xi):  # below we assume <h0_xi>  <=  <xi>
        xi *= -1.
        h0_xi *= -1.

    alpha_thr = np.quantile(h0_xi, 1. - alpha)

    # sort by xi
    idx = np.argsort(xi)
    fractions = fractions[idx]
    xi = xi[idx]

    thr_idx = np.where(xi >= alpha_thr)[0][0]

    fracs = np.array(sorted(list(set(fractions))))

    beta = np.zeros_like(fracs)

    for i_f, f in enumerate(fracs):
        idx = np.where(fractions == fracs[i_f])[0]
        idx_left = np.where(idx < thr_idx)[0]
        beta[i_f] = len(idx_left)/len(idx)

    th_eta = 1.

    if beta_threshold is not None:
        i = np.where(beta < beta_threshold)[0]
        if len(i) > 0:
            th_eta = fracs[i[0]]

    return fracs, beta, th_eta


def calc_beta(gen, model, _alpha, gen2=None, threshold=0., swap_hypotheses=False):
    """
    :param gen: sample generator
    :param model: NN model
    :param alpha: maximal type I error
    :param gen2: if gen2 is not None gen output is used for 0-hypothesis and gen2 for alternative
    otherwise frac > 0 condition is used
    :param swap_hypotheses swap hypotheses H0 and H1 hypotheses
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
                batch_src = batch_frac > threshold
            else:
                batch_src = np.full(len(batch_frac), i == 1)
            if src is None:
                src = batch_src
                xi = batch_xi
                frac = batch_frac
            else:
                src = np.concatenate((src, batch_src))
                xi = np.concatenate((xi, batch_xi))
                frac = np.concatenate((frac, batch_frac))
        g.return_frac = save

    if swap_hypotheses:
        src = np.logical_not(src)

    h0 = np.logical_not(src)  # is 0-hypothesis
    h0_xi = xi[h0]
    xi = xi[src]

    mult = 1.
    if np.mean(h0_xi) > np.mean(xi):  # below we assume <h0_xi>  <=  <xi>
        mult = -1.
        xi *= -1.
        h0_xi *= -1.

    alpha_thr = np.quantile(h0_xi, 1. - _alpha)

    # sort by xi
    xi = np.sort(xi)

    thr_idx = np.where(xi >= alpha_thr)[0][0]
    beta = thr_idx / len(xi)

    return beta, mult * h0_xi, mult * xi


def main():
    import argparse
    import train_healpix
    cline_parser = argparse.ArgumentParser(description='Calculate beta as function of from-source fraction',
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def add_arg(*pargs, **kwargs):
        cline_parser.add_argument(*pargs, **kwargs)

    add_arg('--f_src', type=float, help='fraction of "from-source" EECRs [0,1] or -1 for random', default=-1)
    add_arg('--Neecr', type=int, help='Total number of EECRs in each sample', default=500)
    add_arg('--Emin', type=int, help='Emin in EeV for which the input sample was generated', default=56)
    # add_arg('--source_id', type=str,
    #         help='source (CenA, NGC253, M82, M87 or FornaxA) or comma separated list of sources or "all"',
    #         default='CenA')
    add_arg('sources', type=str, nargs='+', metavar='source', default=[])
    add_arg('--fractions', type=float, nargs='+', metavar='frac',
            help='fractions for mixed source case (in the same order as sources)', default=[])
    add_arg('--Nside', type=int, help='healpix grid Nside parameter', default=32)
    add_arg('--Nini', type=int, help='Size of the initial sample of from-source events', default=10000)
    add_arg('--source_vicinity_radius', type=str, help='source vicinity radius', default='1')
    add_arg('--log_sample', action='store_true', help="sample f_src uniformly in log scale")
    add_arg('--f_src_max', type=float, help='maximal fraction of "from-source" EECRs [0,1]', default=1)
    add_arg('--f_src_min', type=float, help='minimal fraction of "from-source" EECRs [0,1]', default=0)
    add_arg('--models', type=str, nargs='+', metavar='model', help='healpix NN(s)')
    add_arg('--labels', type=str, nargs='+', metavar='label', help='model labels', default=[])
    add_arg('--n_samples', type=int, help='number of samples', default=100000)
    add_arg('--alpha', type=float, nargs='+', metavar='alpha', help='type 1 maximal error', default=[0.01])
    add_arg('--suffix', type=str, default='*')
    add_arg('--batch_size', type=int, help='size of training batch', default=100)
    add_arg('--mf', type=str, help='Magnetic field model (jf or pt)', default='jf')
    add_arg('--data_dir', type=str, help='data root directory (should contain jf/sources/ or pt/sources/)',
            default='data')
    add_arg('--threshold', type=float,
            help='source fraction threshold for binary classification', default=0.0)
    add_arg('--seed', type=int, help='sample generator seed', default=train_healpix.test_seed)
    add_arg('--output', type=str, help='output file name (without extension)', default='beta')
    add_arg('--beta_threshold', type=float, help='threshold beta value for eta output', default=0.05)
    add_arg('--sigmaLnE', type=float, help='deltaE/E energy resolution', default=0.2)
    add_arg('--lgEbin', type=float, help='Log10 energy bin', default=0.05)
    add_arg('--Emax', type=int, help='maximal binning energy in EeV', default=300)
    add_arg('--EminBin', type=float, help='minimal binning energy in EeV', default=56)
    add_arg('--EminSigmaDif', type=float,
            help='minimal difference in between Emin and EminBin in terms of sigma used for param validation',
            default=3)
    add_arg('--disable_energy_binning', action='store_true', help='legacy mode without binning in energy')

    args = cline_parser.parse_args()

    if len(args.fractions) > 0:
        assert len(args.fractions) == len(args.sources) and len(args.sources) > 1

    if len(args.labels) > 0:
        assert len(args.labels) == len(args.models)
        labels = args.labels
    elif len(args.models) > 1:
        labels = [l.split('/')[-1] for l in args.models]
    else:
        labels = ['']

    gen = train_healpix.SampleGenerator(
        args, deterministic=True, sources=args.sources, suffix=args.suffix, seed=args.seed, mixture=args.fractions
    )

    curves = []
    for l, m in zip(labels, args.models):
        model = train_healpix.create_model(gen.Ncells, pretrained=m)

        save_data = {}
        for alpha in args.alpha:
            frac, beta, th_eta = calc_beta_eta(gen, model, alpha, beta_threshold=args.beta_threshold)
            curves.append((l, alpha, frac, beta, th_eta))
            save_data['alpha'] = (frac, beta, th_eta)
            del model
            with open(m + '.eta', mode='a') as out:
                print(th_eta, args.beta_threshold, alpha, args.Neecr, file=out, end='')
                if len(args.sources) == 1:
                    print('\t', *args.sources, file=out)
                else:
                    for f, s in zip(args.fractions, args.sources):
                        print('\t', s, f, file=out, end='')
                    print(file=out)
        outz = args.output
        if len(l) > 0:
            outz += ("_" + l)
        np.savez(outz + '.npz', save_data)
    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    f = None
    for l, alpha, frac, beta, th_eta in curves:
        l = ' '.join([l, r'$\alpha={}$'.format(alpha)])
        i = np.argmin(beta) + 1
        plt.plot(frac[:i], 1. - beta[:i], label=l)
        f = frac[:i]

    plt.plot(f, (1.-args.beta_threshold)*np.ones_like(f), label='', color='black', linestyle='dotted')

    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$1-\beta$')

    plt.legend(loc='lower right')
    plt.savefig(args.output + '.pdf')


if __name__ == '__main__':
    main()
