from abc import ABC, abstractmethod
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np

class Exposure(ABC):
    """
    abstract class for exposure
    """
    @property
    def energy_dependent(self):
        return True

    @abstractmethod
    def eq_exposure(self, ra_deg, dec_deg, energy_EeV=0):
        """
        :param ra_deg: Right ascension in degrees
        :param dec_deg:  Declination in degrees
        :param energy_EeV:  event energy in EeV

        :return: relative exposure calculated for equatorial coordinates
        """
        pass

    def gal_exposure(self, l, b, energy_EeV=0):
        """
        :param l: longitude in degrees
        :param b: latitude in degrees
        :param energy_EeV:  event energy in EeV
        :return: relative exposure calculated for galactic coordinates
        """
        c = SkyCoord(l=l * u.degree, b=b * u.degree, frame='galactic')
        icrs = c.icrs
        ra = icrs.ra/u.degree
        dec = icrs.dec/u.degree
        return self.eq_exposure(ra, dec, energy_EeV=energy_EeV)

class GeometricExposure(Exposure):
    """
    pure geometric exposure at certain latitude implementation
    see page 6 of https://arxiv.org/pdf/astro-ph/0004016.pdf
    """
    def __init__(self, detector_latitude_deg, max_theta_deg):
        """
        :param detector_latitude_deg: geographic latitude of the detector in degrees
        :param max_theta_deg: maximal zenith angle for the observations in degrees
        """
        self._cos_max_theta = np.cos(max_theta_deg * np.pi / 180)
        self._sin_detector_lat = np.sin(detector_latitude_deg * np.pi / 180)
        self._cos_detector_lat = np.cos(detector_latitude_deg * np.pi / 180)

    @property
    def energy_dependent(self):
        return False

    def eq_exposure(self, ra_deg, dec_deg, energy_EeV=0):
        """
        :param ra_deg: vector containing right ascension in degrees
        :param dec_deg: vector containing declination in degrees
        :return: relative exposure vector calculated for equatorial coordinates
        """
        dec = np.array(dec_deg) * np.pi / 180
        cos_dec = np.cos(dec)
        sin_dec = np.sin(dec)

        xi = (self._cos_max_theta - self._sin_detector_lat * sin_dec) / (cos_dec * self._cos_detector_lat)
        xi[xi > 1] = 1
        xi[xi < -1] = -1
        alpha_m = np.arccos(xi)
        exposure = np.sin(alpha_m)*self._cos_detector_lat*cos_dec + alpha_m*self._sin_detector_lat*sin_dec
        return exposure


def create_exposure(args):
    if args.exposure == 'TA':
        return GeometricExposure(detector_latitude_deg=39.2969, max_theta_deg=55)
    elif args.exposure == 'uniform':
        return None
    else:
        assert False, 'unknown exposure'
