import logging

import numpy as np
import healpy as hp

from omegaconf import DictConfig
from sys import stderr
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Generator

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tensorflow.keras.utils import Sequence

from argparse import Namespace
from astropy.coordinates import SkyCoord
from astropy import units as u

from .exposure import create_exposure, Exposure
from .utils import f_sampler, load_src_sample   # TODO: must be independent


class BaseGenerator(ABC):
    """
    Base class for generator.
    """
    def __init__(
            self,
            args: Namespace | DictConfig,
            deterministic: Optional[bool] = None,
            seed: int = 0,
            n_samples: Optional[int] = None,
            return_frac: bool = False,
            suffix: str = '',
            sources: Optional[List[str]] = None,
            mixture: Optional[list] = None,
            add_iso: Optional[bool] = None,
            sampler: Union[str, Generator[tuple[int, int]], int] = "auto",
            batch_size: Optional[int] = None,
            mf: Optional[str] = None
    ) -> None:

        self.point_exposure: list = []
        self.exposure: Exposure = create_exposure(args)
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
        self.logEmin = np.log(args.EminData) if args.EminData is not None else  np.log(args.Emin)

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

        self._batch_size = batch_size if batch_size is not None else args.batch_size
        self.n_batches = int(np.ceil(n_samples / self._batch_size))
        self.n_samples = n_samples

        batch_size = int(np.round(n_samples / self.n_batches))
        if self.add_iso:
            batch_size = (batch_size//2)*2  # make sure batch_size is divisible of 2

        if batch_size != self._batch_size:
            print('batch size adjusted to ', batch_size, file=stderr)
            self._batch_size = batch_size

        self.Neecr = args.Neecr
        self.coordinates = []  # x,y,z,log(E)
        self.source_weights = None

        fE, lnE = self._load_iso_flux(args.data_dir)
        idx = np.where(lnE >= self.logEmin - 3 * self.sigmaLnE)[0]
        self.lnE_iso = lnE[idx]
        self.p_iso = fE[idx]
        self.p_iso /= np.sum(self.p_iso)

        self._setup(mixture=mixture, suffix=suffix, sources=sources, mf=mf)

        self.Nside = args.Nside
        self.threshold = args.threshold

    def _setup(self, mixture:  Union[list, None], suffix: str, sources: list, mf: str):

        if mixture is None:
            mixture = []

        if self.sampler is not None:  # not isotropy
            data_list = list(load_src_sample(self.__args, suffix=suffix, sources=sources, mf=mf))
            if len(mixture) > 0:
                assert len(mixture) == len(data_list), 'inconsistent mixture fractions'
                self.source_weights = np.array(mixture) / np.sum(mixture)

            # 2. Find non-zero lines, i.e., those with Z>0:

            for data in data_list:
                # Filtering invalid entries
                data = data[data[:, 5] > 0]
                if len(data) < self.__args.Neecr:
                    logging.warning('src_sample data size is less then Neecr')
                    # this is just warning since we still can sample with replacement
                if len(data) < self.__args.Neecr//2:
                    assert False, 'src_sample data size is less then Neecr/2'

                l_deg = data[:, 1]
                b_deg = data[:, 0]
                energy = data[:, 6]
                src_cells_file = data[:, 7].astype(np.int32)
                src_cells_cur_grid = hp.ang2pix(
                    self.__args.Nside, l_deg, b_deg, lonlat=True
                )
                if np.sum(src_cells_file != src_cells_cur_grid) > 0:
                    logging.warning(f'healpix grid index check failed. Map will be converted to Nside={self.__args.Nside})')

                l_deg, b_deg = hp.pix2ang(self.__args.Nside, src_cells_cur_grid, lonlat=True)

                c = SkyCoord(l=l_deg * u.degree, b=b_deg * u.degree, frame='galactic')
                xyz = np.array(c.galactic.cartesian.xyz).transpose()
                x4 = np.log(energy).reshape((-1,1))
                coordinates = np.hstack((xyz, x4))  # x,y,z,(E/EeV)^-2
                self.coordinates.append(coordinates)

                if self.exposure is not None:
                    # TODO: take into account exposure energy dependence for iso component
                    assert not self.exposure.energy_dependent, "exposure energy dependece not supported in unbinned mode"

                    points_exposure = self.exposure.gal_exposure(l_deg, b_deg, energy)
                    tot_exposure = np.sum(points_exposure)
                    if tot_exposure == 0:
                        raise ValueError('exposure in the direction of source is equal to 0')

                    n_non_zero_points = np.sum(points_exposure > 0)
                    if n_non_zero_points < self.Neecr:
                        logging.warning(f'number of nonzero exposure points is {n_non_zero_points}')
                    points_exposure /= tot_exposure
                    self.point_exposure.append(points_exposure)

    @staticmethod
    def _load_iso_flux(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
        # TODO: add iso_flux generation
        iso_path = data_dir + '/iso_flux'
        iso_flux = np.loadtxt(iso_path)
        E = iso_flux[:, 0] / 1e18  # EeV
        fE = np.sum(iso_flux[:, 1:], axis=1)/E
        lnE = np.log(E)
        return fE, lnE

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def _generate_single_sample(self, index: int) -> tuple[np.ndarray, float]:
        """
        Method for generating single map.
        """

        if self.sampler is None or (self.add_iso and index % 2 == 0):
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
                log_energies = coordinates[:, 3] + np.random.randn(len(coordinates)) * self.sigmaLnE
                idxs = np.where(log_energies > self.logEmin)[0]

                if len(idxs) < n_src // 2 + 1:
                    assert False, 'too few points to sample from'

                coordinates = coordinates[idxs]

                if self.point_exposure:
                    p = self.point_exposure[file_idx][idxs]
                    p = p / np.sum(p)
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
                p = self.exposure.gal_exposure(lon_iso * 180 / np.pi, lat_iso * 180 / np.pi)
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
            iso_cells = hp.ang2pix(
                self.__args.Nside, np.rad2deg(lon_iso), np.rad2deg(lat_iso), lonlat=True
            )
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
            coordinates[:, 3] = 1000 / (E * E)  # x,y,z, 1000 * (E/EeV)^-2

        answer = Nsrc / self.Neecr

        if not self.return_frac:
            answer = (answer > self.threshold)

        if self.exclude_energy:
            coordinates = coordinates[:, :3]

        return coordinates, answer


class SampleGeneratorTorch(BaseGenerator, Dataset):
    """
    Sample generator for pytorch.Dataset class.
     NOTE: __getitem__ returns a single map

     n_batches:
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:

        if self.deterministic and index == 0:
            np.random.seed(index + self.seed)

        feature, answer = self._generate_single_sample(index=index)

        return torch.from_numpy(feature.astype(np.float32)), torch.tensor(answer, dtype=torch.float32)

    def __len__(self) -> int:
        return self.n_samples


class SampleGeneratorKeras(BaseGenerator, Sequence):
    """
    Sample generator for keras.Sequence
     NOTE: __getitem__ returns batch of maps
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, batch_i: int) -> tuple[np.ndarray, np.ndarray]:
        if self.deterministic and batch_i == 0:
            np.random.seed(batch_i + self.seed)

        answers = []
        batch = []
        for i in range(self.batch_size):
            feature, answer = self._generate_single_sample(index=i)
            answers.append(answer)
            batch.append(feature)

        batch = np.stack(batch, axis=0)
        answers = np.stack(answers, axis=0)
        return batch, answers

    def __len__(self) -> int:
        return self.n_batches
