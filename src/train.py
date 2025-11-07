import losses
from utils import add_arg, create_cline_parser, cl_args


def get_loss(loss):
    # try to find loss by name in losses module
    # if not found use string as is
    try:
        loss = getattr(losses, loss)
    except AttributeError:
        pass
    return loss


def plot_learning_curves(history, save_file=None, show_fig=False):
    import matplotlib.pyplot as plt
    metrics = ['loss'] + [m for m in history.history if m != 'loss' and not m.startswith('val_')]

    for i, m in enumerate(metrics):
        plt.subplot(len(metrics), 1, 1+i)
        labels = []
        if 'val_' + m in history.history:
            plt.plot(history.history['val_' + m], 'r')
            labels.append('Test')
        plt.plot(history.history[m], 'g')
        labels.append('Train')
        plt.legend(labels, loc='upper right')
        plt.ylabel(m)

    if save_file:
        plt.savefig(save_file)

    try:
        if show_fig:
            plt.show()
    except Exception:
        pass


def init_train_cline_args(description):
    init_common_cline_args(description)
    add_arg('--source_id', type=str, help='source (CenA, NGC253, M82, M87 or FornaxA) or comma separated list of sources or "all"',
            default='CenA')
    add_arg('--n_epochs', type=int, help='number of training epochs', default=20)
    add_arg('--show_fig', action='store_true', help="Show learning curves")
    add_arg('--n_early_stop', type=int, help='number of epochs to monitor for early stop', default=10)
    add_arg('--pretrained', type=str, help='pretrained network', default='')
    add_arg('--loss', type=str, help='NN loss', default='binary_crossentropy')
    add_arg('--monitor', type=str, help='NN metrics: used for early stop val_loss/frac', default='frac')
    add_arg('--n_validation_samples', type=int, help='number of samples used for validation after each epoch', default=10000)
    add_arg('--n_test_samples', type=int, help='number of samples used for final test', default=50000)
    add_arg('--min_version', type=int, help='minimal version number for output naming', default=0)
    add_arg('--plot_learning_curves', action='store_true', help='make learning curve plots')
    add_arg('--min_delta', type=float, help='minimal monitor difference used for early stop for loss monitor', default=1e-6)
    add_arg('--evaluate_test_loss', action='store_true', help='evaluate loss and accuracy on test data (minimal fraction is always evaluated)')


def init_common_cline_args(description):
    create_cline_parser(description)
    add_arg('--f_src', type=float, help='fraction of "from-source" EECRs [0,1] or -1 for random', default=-1)
    add_arg('--Neecr', type=int, help='Total number of EECRs in each sample', default=500)
    add_arg('--Emin', type=int, help='Emin in EeV for which the input sample was generated', default=28)
    add_arg('--data_dir', type=str, help='data root directory (should contain [mf]/sources/)',
            default='data')
    add_arg('--mf', type=str, help='Magnetic field model used for training (jf | jf_sol | jf_pl | tf | pt)', default='jf')
    add_arg('--validation_mf', type=str, help='Magnetic field model used for validation (jf | jf_sol | jf_pl | tf | pt)',
            default='jf')
    add_arg('--test_mf', type=str, help='Magnetic field model to use for final test (jf | jf_sol | jf_pl | tf | pt)', default='jf')
    add_arg('--Nside', type=int, help='healpix grid Nside parameter', default=32)
    add_arg('--Nini', type=int, help='Size of the initial sample of from-source events', default=10000)
    add_arg('--log_sample', action='store_true', help="sample f_src uniformly in log scale")
    add_arg('--f_src_max', type=float, help='maximal fraction of "from-source" EECRs [0,1]', default=1)
    add_arg('--f_src_min', type=float, help='minimal fraction of "from-source" EECRs [0,1]', default=0)
    add_arg('--output_prefix', type=str, help='output model file path prefix', default='')
    add_arg('--batch_size', type=int, help='size of training batch', default=256)
    add_arg('--n_samples', type=int, help='number of samples per training epoch', default=10000)
    add_arg('--source_vicinity_radius', type=str, help='source vicinity radius', default='1')
    add_arg('--threshold', type=float,
            help='source fraction threshold for binary classification', default=0.0)
    add_arg('--deterministic', action='store_true', help="use deterministic batches for training (default is random)")
    add_arg('--alpha', type=float, help='type 1 maximal error', default=0.01)
    add_arg('--beta', type=float, help='type 2 maximal error', default=0.05)

    add_arg('--sigmaLnE', type=float, help='deltaE/E energy resolution', default=0.2)
    add_arg('--EminData', type=float, help='minimal data energy in EeV', default=None)

    add_arg('--exposure', type=str, help='exposure: uniform/TA', default='uniform')
    add_arg('--exclude_energy', action='store_true', help='do not include energy into feature list')

