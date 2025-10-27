import sys
import pytest
from os import path
import numpy as np
from argparse import Namespace
from omegaconf import OmegaConf
from src.generators import SampleGeneratorKeras, SampleGeneratorTorch, BaseGenerator

DATA_PATH: str = '../src/data/'

torch_available = 'torch' in sys.modules
keras_available = 'tensorflow' in sys.modules or 'keras' in sys.modules

iso_flux_exists = path.isfile(DATA_PATH + 'iso_flux')


def get_default_args() -> Namespace:
    """Create a Namespace with default values for all command line arguments"""
    return Namespace(
        # From init_common_cline_args
        f_src=-1.0,
        Neecr=500,
        Emin=56,
        data_dir=DATA_PATH,
        mf='jf',
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


class TestableBaseGenerator(BaseGenerator):
    """Concrete implementation for testing BaseGenerator"""

    def __getitem__(self, idx):
        """Implementation for testing"""
        return self._generate_single_sample(idx)

    def __len__(self):
        """Implementation for testing"""
        return self.n_samples or 100



@pytest.mark.skipif(not iso_flux_exists, reason="iso_flux file not found")
class TestBaseGenerator:
    """Tests for the BaseGenerator class"""
    def test_base_generator_initialization_namespace(self):
        """Test that BaseGenerator can be initialized with required parameters"""
        generator = TestableBaseGenerator(
            args=ARGS,
            deterministic=True,
            seed=42,
            batch_size=32,
            return_frac=False, suffix=''
        )

        assert generator.deterministic is True
        assert generator.seed == 42
        assert generator.batch_size == 32
        assert generator.return_frac is False

        generator = TestableBaseGenerator(
            args=ARGS,
            deterministic=True,
            seed=42,
            batch_size=33,
            return_frac=False, suffix=''
        )
        assert generator.batch_size % 2 == 0,  "batch_size must be adjusted"

    def test_base_generator_initialization_yaml(self):
        """Test that BaseGenerator can be initialized with parameters from yaml"""
        config = OmegaConf.load('../config/main.yaml')
        config.data['data_dir'] = DATA_PATH

        generator = TestableBaseGenerator(
            args=config.data,
            deterministic=True,
            seed=42,
            batch_size=32,
            return_frac=False, suffix=''
        )

        assert generator.deterministic is True
        assert generator.seed == 42
        assert generator.return_frac is False

        generator = TestableBaseGenerator(
            args=config.data,
            deterministic=True,
            seed=42,
            batch_size=33,
            return_frac=False, suffix=''
        )
        assert generator.batch_size % 2 == 0,  "batch_size must be adjusted"


@pytest.mark.skipif(not iso_flux_exists, reason="iso_flux file is not found")
class TestTorchGenerator:
    """Tests for TorchGenerator class (if PyTorch is available)"""

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_generator_initialization(self):
        """Test TorchGenerator initialization"""
        import torch

        generator = SampleGeneratorTorch(
            args=ARGS,
            batch_size=32,
            deterministic=True,
            seed=42
        )

        assert isinstance(generator, torch.utils.data.Dataset)
        assert generator.batch_size == 32

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_generator_getitem_returns_tensors(self):
        """Test that __getitem__ returns PyTorch tensors"""
        import torch

        generator = SampleGeneratorTorch(args=ARGS, batch_size=32, return_frac=True)
        features, answer = generator[0]

        assert isinstance(features, torch.Tensor)
        assert isinstance(answer, torch.Tensor)
        assert features.dtype == torch.float32
        assert answer.dtype == torch.float32
        assert features.shape[0] == ARGS.Neecr

    @pytest.mark.skipif(not torch_available, reason="PyTorch not available")
    def test_torch_generator_with_dataloader(self):
        """Test integration with PyTorch DataLoader"""
        import torch
        from torch.utils.data import DataLoader

        batch_size = 4
        generator = SampleGeneratorTorch(args=ARGS, batch_size=batch_size, n_samples=12)
        dataloader = DataLoader(generator, batch_size=generator.batch_size, shuffle=False)

        for batch_features, batch_answers in dataloader:
            assert isinstance(batch_features, torch.Tensor)
            assert isinstance(batch_answers, torch.Tensor)
            assert batch_features.shape[0] == 4
            assert batch_answers.shape[0] == 4
            assert batch_features.shape[1] == ARGS.Neecr



@pytest.mark.skipif(not iso_flux_exists, reason="iso_flux file is not found")
class TestKerasGenerator:
    """Tests for KerasGenerator class (if Keras/TensorFlow is available)"""

    @pytest.mark.skipif(not keras_available, reason="Keras not available")
    def test_keras_generator_initialization(self):
        """Test KerasGenerator initialization"""
        from tensorflow.keras.utils import Sequence

        generator = SampleGeneratorKeras(
            args=ARGS,
            batch_size=32,
            deterministic=True,
            seed=42
        )

        assert isinstance(generator, Sequence)
        assert generator.batch_size == 32

    @pytest.mark.skipif(not keras_available, reason="Keras not available")
    def test_keras_generator_getitem_returns_arrays(self):
        """Test that __getitem__ returns numpy arrays"""
        generator = SampleGeneratorKeras(args=ARGS, batch_size=4, return_frac=True)
        batch_features, batch_answers = generator[0]

        assert isinstance(batch_features, np.ndarray)
        assert isinstance(batch_answers, np.ndarray)
        assert batch_features.shape[0] == 4
        assert batch_answers.shape[0] == 4
        assert batch_features.shape[1] == ARGS.Neecr

        generator = SampleGeneratorKeras(args=ARGS, batch_size=3, return_frac=True)
        batch_features, batch_answers = generator[0]

        assert batch_features.shape[0] % 2 == 0, "batch_size must be divisible of 2"

    @pytest.mark.skipif(not keras_available, reason="Keras not available")
    def test_keras_generator_length(self):
        """Test that __len__ returns correct number of batches"""
        generator = SampleGeneratorKeras(args=ARGS, batch_size=32, n_samples=100)

        # n_samples=100, batch_size=32 → ceil(100/32)=4 batches
        assert len(generator) == 4

