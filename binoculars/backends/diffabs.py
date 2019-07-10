'''This file is part of the binoculars project.

  The BINoculars library is free software: you can redistribute it
  and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, either version 3 of
  the License, or (at your option) any later version.

  The BINoculars library is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty
  of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the hkl library.  If not, see
  <http://www.gnu.org/licenses/>.

  Copyright (C) 2015-2019 Synchrotron SOLEIL
                          L'Orme des Merisiers Saint-Aubin
                          BP 48 91192 GIF-sur-YVETTE CEDEX

  Copyright (C) 2012-2015 European Synchrotron Radiation Facility
                          Grenoble, France

  Authors: Willem Onderwaater <onderwaa@esrf.fr>
           Picca Frédéric-Emmanuel <picca@synchrotron-soleil.fr>

'''
from typing import Dict, Namedtuple, Optional, Tuple

import numpy
import math
import os

from gi.repository import Hkl
from h5py import Dataset, File
from numpy import ndarray
from pyFAI.detectors import ALL_DETECTORS

from .soleil import (HItem,
                     get_nxclass)

# put hkl_matrix_to_numpy into soleil
from .sixs import Dataframe, M, PDataFrame, SIXS, WRONG_ATTENUATION,\
    hkl_matrix_to_numpy
from ..util import status

##################
# Input Backends #
##################


class Values(Namedtuple):
    image: ndarray
    attenuation: Optional[float]
    timestamp: float
    values: Tuple[float]


class Diffabs(SIXS):
    HPATH = {
        "image": HItem("data_02", False),
        "mu": HItem("data_16", False),
        "komega": HItem("data_16", False),
        "kappa": HItem("data_16", False),
        "kphi": HItem("data_16", False),
        "gamma": HItem("data_18", False),
        "delta": HItem("data_17", False),
        "timestamp": HItem("sensors_timestamps", True),
    }

    def get_pointcount(self, scanno: int) -> int:
        # just open the file in order to extract the number of step
        with File(self.get_filename(scanno), 'r') as scan:
            return get_nxclass(scan, "NXdata")['data_03'].shape[0]

    def get_attenuation(self,
                        index: int,
                        h5_nodes: Dict[str, Dataset],
                        offset: int) -> Optional[float]:
        attenuation = None
        if self.config.attenuation_coefficient is not None:
            try:
                try:
                    node = h5_nodes['attenuation']
                    if node is not None:
                        attenuation = node[index + offset]
                    else:
                        raise Exception("you asked for attenuation but the file does not contain attenuation informations.")  # noqa
                except KeyError:
                    attenuation = 1.0
            except ValueError:
                attenuation = WRONG_ATTENUATION
        return attenuation

    def get_timestamp(self, index: int, h5_nodes: Dict[str, Dataset]) -> float:
        timestamp = None
        try:
            timestamp = h5_nodes['timestamp'][index]
        except KeyError:
            timestamp = index
        return timestamp

    def get_values(self, index: int, h5_nodes: Dict[str, Dataset]) -> Values:
        image = h5_nodes['image'][index]
        mu = h5_nodes['mu'][index]
        komega = h5_nodes['komega'][index]
        kappa = h5_nodes['kappa'][index]
        kphi = h5_nodes['kphi'][index]
        gamma = h5_nodes['gamma'][index]
        delta = h5_nodes['delta'][index]
        timestamp = self.get_timestamp(index, h5_nodes)

        return Values(image, None, timestamp,
                      (mu, komega, kappa, kphi, gamma, delta))

    def process_image(self, index: int, dataframe: Dataframe,
                      pixels: ndarray,
                      mask: ndarray) -> Tuple[ndarray, ndarray,
                                              Tuple[int, PDataFrame]]:
        status(str(index))

        # extract the data from the h5 nodes
        h5_nodes = dataframe.h5_nodes
        intensity, attenuation, timestamp, values = self.get_values(index, h5_nodes)

        # BEWARE in order to avoid precision problem we convert the
        # uint16 -> float32. (the size of the mantis is on 23 bits)
        # enought to contain the uint16. If one day we use uint32, it
        # should be necessary to convert into float64.
        intensity = intensity.astype('float32')

        weights = None
        if self.config.attenuation_coefficient is not None:
            if attenuation != WRONG_ATTENUATION:
                intensity *= self.config.attenuation_coefficient ** attenuation
                weights = numpy.ones_like(intensity)
                weights *= ~mask
            else:
                weights = numpy.zeros_like(intensity)
        else:
            weights = numpy.ones_like(intensity)
            weights *= ~mask

        k = 2 * math.pi / dataframe.source.wavelength

        hkl_geometry = dataframe.diffractometer.geometry
        hkl_geometry.axis_values_set(values, Hkl.UnitEnum.USER)

        # sample
        hkl_sample = dataframe.sample.sample
        q_sample = hkl_geometry.sample_rotation_get(hkl_sample)
        R = hkl_matrix_to_numpy(q_sample.to_matrix())

        # detector
        hkl_detector = dataframe.detector.detector
        q_detector = hkl_geometry.detector_rotation_get(hkl_detector)
        P = hkl_matrix_to_numpy(q_detector.to_matrix())

        if self.config.detrot is not None:
            P = numpy.dot(P, M(math.radians(self.config.detrot), [1, 0, 0]))

        surface_orientation = self.config.surface_orientation

        pdataframe = PDataFrame(pixels, k, dataframe.diffractometer.ub, R, P, index, timestamp, surface_orientation)

        return intensity, weights, (index, pdataframe)

    def get_pixels(self, detector):
        # works only for flat detector.
        detector = ALL_DETECTORS[detector.name]()
        y, x, _ = detector.calc_cartesian_positions()
        y0 = y[self.config.centralpixel[1], self.config.centralpixel[0]]
        x0 = x[self.config.centralpixel[1], self.config.centralpixel[0]]
        z = numpy.ones(x.shape) * -1 * self.config.sdd
        # return converted to the hkl library coordinates
        # x -> -y
        # y -> z
        # z -> -x
        return numpy.array([-z, -(x - x0), (y - y0)])


def load_matrix(filename: str) -> Optional[ndarray]:
    if filename is None:
        return None
    if os.path.exists(filename):
        ext = os.path.splitext(filename)[-1]
        if ext == '.txt':
            return numpy.array(numpy.loadtxt(filename), dtype=numpy.bool)
        elif ext == '.npy':
            mask = numpy.array(numpy.load(filename), dtype=numpy.bool)
            print("loaded mask sum: ", numpy.sum(mask))
            return mask
        else:
            raise ValueError('unknown extension {0}, unable to load matrix!\n'.format(ext))  # noqa
    else:
        raise IOError('filename: {0} does not exist. Can not load matrix'.format(filename))  # noqa
