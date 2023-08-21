import argparse
import numpy as np
import logging
from ray.tune import Trainable as BaseTrainable
from os import path, mkdir, listdir
from ray.tune.analysis.experiment_analysis import Analysis
from ray.tune.utils import pin_in_object_store, get_pinned_object
from train_gcnn import source_data, SampleGenerator, train_seed, val_seed, get_loss, test_seed
from gcnn import create_model
from tensorflow import keras

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

STOP = 'done'

class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

class Trainable(BaseTrainable):
    args_id = None

    @classmethod
    def _convert_config(cls, config: dict):
        # support multilayer hyperparam declarations in hyperopt
        dict_items = [i for i in config.values() if type(i) == dict]
        if len(dict_items) == 0:
            return config
        result = config.copy()
        for i in dict_items:
            result.update(i)
        return cls._convert_config(result)

    def setup(self, config):
        cline_args = get_pinned_object(self.args_id)
        assert cline_args != None, 'Failed to pass command line params'

        args = objdict(cline_args.__dict__)
        config = self._convert_config(config)
        args.update(config)  # this can override some command line params
        self._config = args

        if args.source_id == 'all':
            args.source_id = ','.join(sorted(source_data.keys()))

        args.loss = get_loss(args.loss)

        self.train_gen = SampleGenerator(args, seed=train_seed)
        train_mf = args.mf
        args.mf = train_mf if args.validation_mf == 'same' else args.validation_mf
        self.val_gen = SampleGenerator(args, deterministic=True, seed=val_seed, n_samples=args.n_val_samples)
        args.mf = train_mf if args.test_mf == 'same' else args.test_mf
        self.test_gen = SampleGenerator(args, deterministic=True, seed=test_seed, n_samples=args.n_val_samples)
        args.mf = train_mf
        args.n_kernel_layers = int(args.n_kernel_layers)
        args.n_dense_layers = int(args.n_dense_layers)
        args.kernel_layer_size = int(args.kernel_layer_size)
        args.dense_layer_size = int(args.dense_layer_size)
        kernel_layers = args.n_kernel_layers * [args.kernel_layer_size]
        dense_layers = args.n_dense_layers * [args.dense_layer_size]
        n_features = 3 if args.exclude_energy else 4
        dinamic_conv = bool(args.dinamic_conv)

        self._model = create_model(args.Neecr, n_features=n_features, kernel_layers=kernel_layers,
                                   n_conv=int(args.n_convolutions), dense_layers=dense_layers,
                                   k_neighbors=int(args.k_neighbors), pretrained=args.pretrained, l2=args.l2,
                                   l1=args.l1, dropout_rate=args.dropout_rate, normalize=True, activation=args.activation,
                                   output_activation='sigmoid', loss='binary_crossentropy', metrics='accuracy', lr=args.lr,
                                   dinamic_conv=dinamic_conv)

        self.history = []
        self.n_early_stop = args.get('n_early_stop', 0)
        self.early_stop_metric = args.get('monitor', 'val_loss')
        early_stop_mode = args.get('monitor_mode', 'min')
        assert early_stop_mode in ('min', 'max')
        self.early_stop_metric_mult = -1. if early_stop_mode == "min" else 1.
        self.model_file_name = 'model.h5'

    def reset_config(self, new_config):
        from tensorflow.keras.backend import clear_session
        del self._model
        clear_session()
        self._setup(new_config)
        return True

    def step(self):  # This is called iteratively.
        from beta import calc_beta_eta
        result = {}

        history = self._model.fit(self.train_gen, epochs=1, verbose=0, validation_data=self.val_gen)
        for key, val in history.history.items():
            result[key] = val[0]
        alpha = 0.01
        beta = 0.05
        _, _, th_eta = calc_beta_eta(self.val_gen, self._model, alpha, beta_threshold=beta)

        result['frac'] = th_eta

        if self.n_early_stop > 0:
            if self.early_stop_metric not in result:
                self.n_early_stop = 0
                logging.warning('early stop disabled: metric {} not found'.format(self.early_stop_metric))
            else:
                self.history.append(self.early_stop_metric_mult * result[self.early_stop_metric])
                if np.argmax(self.history) == len(self.history)-1:  # best metric so far
                    result.update(should_checkpoint=True)
                else:
                    result['test_frac'] = 100  # skip test
                    result['train_frac'] = 100

        if 'test_frac' not in result:
            _, _, th_eta = calc_beta_eta(self.test_gen, self._model, alpha, beta_threshold=beta)
            result['test_frac'] = th_eta

        if 'train_frac' not in result:
            if self._config.validation_mf not in ['same', self._config.mf]:
                if self._config.test_mf not in ['same', self._config.mf]:
                    from train_healpix import val_seed, train_seed
                    deterministic = self.train_gen.deterministic
                    self.train_gen.seed = val_seed
                    self.train_gen.deterministic = True
                    _, _, th_eta = calc_beta_eta(self.train_gen, self._model, alpha, beta_threshold=beta)
                    result['train_frac'] = th_eta
                    self.train_gen.deterministic = deterministic
                    self.train_gen.seed = train_seed
                else:
                    result['train_frac'] = result['test_frac']
        else:
            result['train_frac'] = result['frac']

        if self.n_early_stop > 0 and len(self.history) - np.argmax(self.history) - 1 >= self.n_early_stop:
            result['done'] = True

        return result

    def _export_model(self, export_formats, export_dir):
        from tensorflow.keras.models import save_model
        print('_export_model', export_dir, *export_formats)
        if 'h5' in export_formats:
            model_path = path.join(export_dir, self.model_file_name)
            save_model(self._model, model_path)
            return {'h5': export_dir}
        return {}

    def save_checkpoint(self, tmp_checkpoint_dir):
        from tensorflow.keras.models import save_model
        max_checkpoints = 1000
        model_dir = path.join(tmp_checkpoint_dir, "mcheckpoint")
        for i in range(max_checkpoints):
            d = model_dir + str(i)
            if not path.exists(d):
                mkdir(d)
                model_dir = d
                break
        if i-1 == max_checkpoints:
            raise Exception('maximal number of checkpoints {} reached'.format(max_checkpoints))

        model_path = path.join(model_dir, self.model_file_name)
        save_model(self._model, model_path)
        return model_dir

    def load_checkpoint(self, model_dir):
        from tensorflow.keras.backend import clear_session
        from tensorflow.keras.models import load_model

        del self._model
        clear_session()
        model_path = path.join(model_dir, self.model_file_name)
        self._model = load_model(model_path)


class Main(object):
    def __init__(self, name='nn_aniso'):
        self.__parser = argparse.ArgumentParser()
        def add_arg(arg_name, **kwargs):
            self.__parser.add_argument(arg_name, **kwargs)

        add_arg('--f_src', type=float, help='fraction of "from-source" EECRs [0,1] or -1 for random', default=-1)
        add_arg('--Neecr', type=int, help='Total number of EECRs in each sample', default=500)
        add_arg('--Emin', type=int, help='Emin in EeV for which the input sample was generated', default=28)
        add_arg('--EminData', type=float, help='minimal data energy in EeV', default=56)
        # add_arg('--Emax', type=int, help='maximal binning energy in EeV', default=300)
        add_arg('--sigmaLnE', type=float, help='deltaE/E energy resolution', default=0.2)
        add_arg('--exclude_energy', action='store_true', help='legacy mode without binning in energy')
        add_arg('--source_id', type=str, help='source (CenA, NGC253, M82, M87 or FornaxA) or comma separated list of sources or "all"', default='CenA')
        add_arg('--data_dir', type=str, help='data root directory (should contain jf/sources/ or pt/sources/)', default='data')
        add_arg('--mf', type=str, help='Magnetic field model (jf/pt/jf_pl/jf_sol/tf)', default='jf')
        add_arg('--validation_mf', type=str, help='Magnetic field model map to use for validation', default='jf_pl')
        add_arg('--test_mf', type=str, help='Magnetic field model map to use for final test', default='jf_sol')
        add_arg('--Nside', type=int, help='healpix grid Nside parameter', default=32)
        add_arg('--Nini', type=int, help='Size of the initial sample of from-source events', default=10000)
        add_arg('--log_sample', action='store_true', help="sample f_src uniformly in log scale")
        add_arg('--f_src_max', type=float, help='maximal fraction of "from-source" EECRs [0,1]', default=1)
        add_arg('--f_src_min', type=float, help='minimal fraction of "from-source" EECRs [0,1]', default=0)
        add_arg('--output_prefix', type=str, help='output model file path prefix', default='')
        add_arg('--batch_size', type=int, help='size of training batch', default=128)
        add_arg('--n_epochs', type=int, help='number of training epochs', default=20)
        add_arg('--n_early_stop', type=int, help='number of epochs to monitor for early stop', default=10)
        add_arg('--pretrained', type=str, help='pretrained network', default='')
        add_arg('--loss', type=str, help='NN loss', default='binary_crossentropy')
        add_arg('--monitor', type=str, help='NN metrics: used for early stop val_loss/frac', default='frac')
        add_arg('--monitor_mode', type=str, help='Best NN metrics: min/max', default='min')
        add_arg('--n_samples', type=int, help='number of samples per training epoch', default=10000)
        add_arg('--n_val_samples', type=int, help='number of validation samples per training epoch', default=10000)
        add_arg('--source_vicinity_radius', type=str, help='source vicinity radius', default='1')
        add_arg('--threshold', type=float, help='source fraction threshold for binary classification', default=0.0)
        add_arg('--deterministic', action='store_true', help="use deterministic batches for training (default is random)")
        add_arg('--alpha', type=float, help='type 1 maximal error', default=0.01)
        add_arg('--beta', type=float, help='type 2 maximal error', default=0.05)
        add_arg('--min_version', type=int, help='minimal version number for output naming', default=0)
        add_arg('--exposure', type=str, help='exposure: uniform/TA', default='uniform')

        add_arg('--lr', type=float, help='learning rate', default=0.001)
        add_arg("--debug", action="store_true", help="Finish quickly for testing")
        add_arg('--max_concurrent', type=int, help='maximal number of concurrent trials', default=2)
        add_arg('--experiment_name', type=str, help='result name', default=name)
        add_arg('--address', type=str, help='ray cluster address', default='')
        add_arg('--print_results', action='store_true', help="print current experiment results summary")
        add_arg('--fixed_config_json', type=str, help='read fixed hyperparams from given json', default='')
        add_arg('--n_attempts', type=int, help='maximal number of metaparam configurations to try', default=100)

    @property
    def _metric(self):
        return dict(metric=self.args.monitor, mode=self.args.monitor_mode)

    @property
    def _meta_params(self):
        if self.args.fixed_config_json:
            import json
            with open(self.args.fixed_config_json) as json_file:
                search_space = json.load(json_file)
        else:
            from hyperopt import hp
            search_space = {
                "n_kernel_layers": hp.quniform("n_kernel_layers", 2, 5, 1),
                "n_convolutions": hp.quniform("n_convolutions", 2, 5, 1),
                "k_neighbors": hp.quniform("k_neighbors", 4, 16, 4),
                "kernel_layer_size": hp.qloguniform("kernel_layer_size", np.log(16), np.log(128), 4),
                "n_dense_layers": hp.quniform("n_dense_layers", 0, 2, 1),
                "dense_layer_size": hp.qloguniform("dense_layer_size", np.log(16), np.log(512), 4),
                "l1": hp.loguniform("l1", np.log(1e-5), np.log(1)),
                "l2": hp.loguniform("l2", np.log(1e-5), np.log(1)),
                "dropout_rate": hp.uniform("dropout_rate", 0, 0.2),
                "activation": hp.choice("activation", ['relu', 'prelu']),
                "dinamic_conv": hp.choice("dinamic_conv", [True, False]),
            }

        return search_space

    @property
    def _resources_per_trial(self):
        return {
                "cpu": 1,
                "gpu": 1
            }


    def _result_report_from_dir(self):
        if not self.args.experiment_name:
            print('--experiment_name param must be specified')

        results_dir = path.expanduser('~/ray_results/' + self.args.experiment_name)
        if not path.isdir(results_dir):
            print(results_dir, 'not found. Make sure to run this command on head node under ray user')
            return

        from ray.tune.analysis.experiment_analysis import Analysis
        analysis = Analysis(results_dir)
        self._result_report(analysis)

    def _result_report(self, analysis: Analysis):
        # Get the best hyperparameters
        best_hyperparameters = analysis.get_best_config(**self._metric)
        for key, val in best_hyperparameters.items():
            print(key, val)
        log_dir = analysis.get_best_logdir(**self._metric)
        print('best run log dir:\n', log_dir)
        df = analysis.dataframe(**self._metric)  # sorting here doesn't work for some reason
        ascending = (self._metric['mode'] == 'min')
        df.sort_values(by=[self._metric['metric']], inplace=True, ascending=ascending)  # sort again
        html_report = self.args.experiment_name + '_report.html'
        csv_report = self.args.experiment_name + '_report.csv'
        with open(html_report, mode='w') as out:
            out.write(df.to_html())
        df.to_csv(csv_report, index=False, header=True, sep='\t')
        print('Full report saved to', html_report, 'and', csv_report)
        best_report = self.args.experiment_name + '_best.txt'
        with open(best_report, mode='w') as out:
            print(df.iloc[0], file=out)
        print('Best run details saved to', best_report)
        training_iteration = df.iloc[0].training_iteration
        print('Best model dir\n{}/checkpoint_{}'.format(log_dir,training_iteration))



    def run(self):
        import ray
        from ray import tune
        self.args, _ = self.__parser.parse_known_args()

        if self.args.print_results:
            self._result_report_from_dir()
            return

        if not self.args.data_dir.startswith('/'):
            from os import getcwd
            self.args.data_dir = getcwd() + '/' + self.args.data_dir

        assert self.args.monitor in ['val_loss','frac']

        import tensorflow as tf

        if 'gpu' in self._resources_per_trial:
            # make sure gpu is detected
            print('is_gpu_available', tf.test.is_gpu_available())

        if self.args.address and not self.args.debug:
            ray.init(address=self.args.address)
        else:
            ray.shutdown()
            ray.init(local_mode=self.args.debug)

        Trainable.args_id = pin_in_object_store(self.args)

        meta_par = self._meta_params

        if type(meta_par) != dict:
            meta_par = None  # passed via _scheduler

        n_epochs = 3 if self.args.debug else self.args.n_epochs

        grace_period = self.args.n_early_stop

        from ray.tune.schedulers import AsyncHyperBandScheduler
        scheduler = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            max_t=self.args.n_epochs,
            grace_period=grace_period,
            **self._metric)

        if self.args.fixed_config_json:
            search_alg = None
        else:
            from ray.tune.suggest.hyperopt import HyperOptSearch
            search_alg = HyperOptSearch(
                self._meta_params,
                **self._metric
            )

        analysis = tune.run(
            Trainable,
            name=self.args.experiment_name,
            scheduler=scheduler,
            search_alg=search_alg,
            stop={"training_iteration": n_epochs},
            num_samples=(1 if self.args.debug else self.args.n_attempts),
            resources_per_trial=self._resources_per_trial,
            config=meta_par,
            reuse_actors=True,
            export_formats=['h5'],
            sync_to_driver=False, # sync folders manually if needed
            checkpoint_freq=1,
        )

        self._result_report(analysis)



if __name__ == "__main__":
    o = Main()
    o.run()
