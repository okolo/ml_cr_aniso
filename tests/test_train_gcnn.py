import os
import time
import numpy as np

from argparse import Namespace
from src.models import create_model, get_model_name, custom_objects


def get_default_args() -> Namespace:
    """Create a Namespace with default values for all command line arguments"""
    return Namespace(
        # From init_common_cline_args
        f_src=-1.0,
        Neecr=30,
        Emin=56,
        data_dir='absolute_path_to/your/project/src/data/',  # absolute path is hardcoded TODO: implement test.resources
        mf='jf',
        use_energy_as_feature=True,
        disable_dinamic_conv=True,
        validation_mf='jf',
        test_mf='jf',
        Nside=32,
        Nini=10000,
        log_sample=False,
        f_src_max=1.0,
        f_src_min=0.0,
        output_prefix='',
        batch_size=256,
        n_samples=10000,
        source_vicinity_radius='1',
        threshold=0.0,
        deterministic=False,
        alpha=0.01,
        beta=0.05,
        sigmaLnE=0.2,
        EminData=None,
        exposure='TA',
        exclude_energy=False,
        # From init_train_cline_args
        source_id='CenA',
        n_epochs=20,
        show_fig=False,
        n_early_stop=10,
        pretrained='',
        loss='binary_crossentropy',
        monitor='frac',
        n_validation_samples=10000,
        n_test_samples=50000,
        min_version=0,
        plot_learning_curves=False,
        min_delta=1e-6,
        evaluate_test_loss=False
    )


ARGS: Namespace = get_default_args()


class TestModel:
    def test_create_model(self):
        n_features = 3 if ARGS.exclude_energy else 4

        model = create_model(
            ARGS.Neecr, n_features=n_features, pretrained=ARGS.pretrained,
            dinamic_conv=(not ARGS.disable_dinamic_conv)
        )

        features = np.random.random((10, ARGS.Neecr, 4)).astype(np.float32)
        answers = np.random.random((10, 1)).astype(np.float32)

        assert model.input_shape == (None, ARGS.Neecr, n_features), "Wrong input shape"
        assert model(features).shape == answers.shape, "Wrong output shape"

    def test_model_save_and_load(self):
        """
        This function checks weights' saving and loading from file.

        Sometimes keras fails to initialize model due to the usage of custom objects,
        thus it's better to doublecheck that changes in models won't affect the procedure
        """
        import tensorflow as tf

        n_features = 3 if ARGS.exclude_energy else 4

        model = create_model(
            ARGS.Neecr, n_features=n_features, pretrained=ARGS.pretrained,
            dinamic_conv=(not ARGS.disable_dinamic_conv)
        )

        save_path = './tmp_' + get_model_name(ARGS) + '.weights.h5'
        model.save(save_path)

        loaded = tf.keras.models.load_model(save_path, custom_objects=custom_objects)
        os.remove(save_path)

        assert model.get_config() == loaded.get_config(), "Different configuration"

        are_equal = True
        if len(model.get_weights()) != len(loaded.get_weights()):
            are_equal = False
        else:
            for w1, w2 in zip(model.get_weights(), loaded.get_weights()):
                if not np.array_equal(w1, w2):
                    are_equal = False
                    break

        assert are_equal, "Different weights"


class TestTraining:
    def test_run_gcnn_training(self):
        """
        Test running the training pipeline on synthetic examples.
        """
        n_features = 3 if ARGS.exclude_energy else 4

        model = create_model(ARGS.Neecr, n_features=n_features, pretrained=ARGS.pretrained,
                             dinamic_conv=(not ARGS.disable_dinamic_conv))

        features = np.random.random((10, ARGS.Neecr, 4)).astype(np.float32)
        answers = np.random.random((10, 1)).astype(np.float32)

        # Test fit with synthetic data
        t = time.time()
        model.fit(
            features, answers,
            batch_size=2,
            epochs=1,
            validation_split=0.2,
            verbose=1
        )
        tf = time.time()

        print(f'Training took {tf - t:.2f} seconds')



