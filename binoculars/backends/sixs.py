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
from typing import Dict, NamedTuple, Optional, Tuple

import numpy
import math
import os
import tables
import sys

from collections import namedtuple
from enum import Enum
from math import cos, sin
from numpy import ndarray
from numpy.linalg import inv
from pyFAI.detectors import ALL_DETECTORS
from gi.repository import Hkl

from .. import backend, errors, util
from ..util import as_string
from tables import Node
from tables.exceptions import NoSuchNodeError

# TODO
# - Angles delta gamma. nom de 2 ou 3 moteurs. omega puis delta
#   gamma pour chaque pixels.

# - aller cherche dans le fichier NeXuS le x0, y0 ainsi que le sdd.

# - travailler en qx qy qz, il faut rajouter un paramètre optionnel
# - qui permet de choisir une rotation azimuthal de Qx Qy.

###############
# Projections #
###############

SurfaceOrientation = Enum("SurfaceOrientation", "VERTICAL HORIZONTAL")

PDataFrame = NamedTuple("PDataFrame", [("pixels", ndarray),
                                       ("k", float),
                                       ("ub", Optional[ndarray]),
                                       ("R", ndarray),
                                       ("P", ndarray),
                                       ("index", int),
                                       ("timestamp", int),
                                       ("surface_orientation", SurfaceOrientation)])


class RealSpace(backend.ProjectionBase):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        pixels = numpy.tensordot(pdataframe.P,
                                 pdataframe.pixels, axes=1)
        x = pixels[1]
        y = pixels[2]
        z = numpy.ones_like(x) * pdataframe.timestamp

        return (x, y, z)

    def get_axis_labels(self):
        return ('x', 'y', 'z')


class Pixels(backend.ProjectionBase):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        return numpy.meshgrid(numpy.arange(pdataframe.pixels[0].shape[1]),
                              numpy.arange(pdataframe.pixels[0].shape[0]))

    def get_axis_labels(self) -> Tuple[str]:
        return 'x', 'y'


class HKLProjection(backend.ProjectionBase):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        pixels, k, UB, R, P, idx, timestamp, _surface_orientation = pdataframe

        if UB is None:
            raise Exception("In order to compute the HKL projection, you need a valid ub matrix")

        ki = [1, 0, 0]
        RUB_1 = inv(numpy.dot(R, UB))
        RUB_1P = numpy.dot(RUB_1, P)
        kf = normalized(pixels, axis=0)
        hkl_f = numpy.tensordot(RUB_1P, kf, axes=1)
        hkl_i = numpy.dot(RUB_1, ki)
        hkl = hkl_f - hkl_i[:, numpy.newaxis, numpy.newaxis]

        h, k, l = hkl * k

        return h, k, l

    def get_axis_labels(self) -> Tuple[str]:
        return 'H', 'K', 'L'


class HKProjection(HKLProjection):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        h, k, l = super(HKProjection, self).project(index, pdataframe)
        return h, k

    def get_axis_labels(self) -> Tuple[str]:
        return 'H', 'K'


class QxQyQzProjection(backend.ProjectionBase):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        # put the detector at the right position

        pixels, k, _UB, R, P, idx, timestamp, surface_orientation = pdataframe

        # TODO factorize with HklProjection. Here a trick in order to
        # compute Qx Qy Qz in the omega basis.
        if surface_orientation is SurfaceOrientation.VERTICAL:
            UB = numpy.array([[1,  0, 0],
                              [0,  0, 1],
                              [0, -1, 0]])

            if self.config.omega_offset is not None:
                UB = numpy.dot(UB,
                               M(self.config.omega_offset, [0, 0, -1]))
        else:
            UB = numpy.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])

            if self.config.mu_offset is not None:
                UB = numpy.dot(UB,
                               M(self.config.mu_offset, [0, 0, -1]))

            
        # the ki vector should be in the NexusFile or easily extracted
        # from the hkl library.
        ki = [1, 0, 0]
        RUB_1 = inv(numpy.dot(R, UB))
        RUB_1P = numpy.dot(RUB_1, P)
        kf = normalized(pixels, axis=0)
        hkl_f = numpy.tensordot(RUB_1P, kf, axes=1)
        hkl_i = numpy.dot(RUB_1, ki)
        hkl = hkl_f - hkl_i[:, numpy.newaxis, numpy.newaxis]

        qx, qy, qz = hkl * k
        return qx, qy, qz

    def get_axis_labels(self) -> Tuple[str]:
        return "Qx", "Qy", "Qz"

    def parse_config(self, config) -> None:
        super(QxQyQzProjection, self).parse_config(config)

        # omega offset for the sample in degree then convert into radian
        omega_offset = config.pop('omega_offset', None)
        if omega_offset is not None:
            self.config.omega_offset = math.radians(float(omega_offset))
        else:
            self.config.omega_offset = None

        # omega offset for the sample in degree then convert into radian
        mu_offset = config.pop('mu_offset', None)
        if mu_offset is not None:
            self.config.mu_offset = math.radians(float(mu_offset))
        else:
            self.config.mu_offset = None

class QxQyIndexProjection(QxQyQzProjection):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        qx, qy, qz = super(QxQyIndexProjection, self).project(index, pdataframe)
        return qx, qy, numpy.ones_like(qx) * pdataframe.timestamp

    def get_axis_labels(self) -> Tuple[str]:
        return 'Qx', 'Qy', 't'


class QxQzIndexProjection(QxQyQzProjection):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        qx, qy, qz = super(QxQzIndexProjection, self).project(index, pdataframe)
        return qx, qz, numpy.ones_like(qx) * pdataframe.timestamp

    def get_axis_labels(self) -> Tuple[str]:
        return 'Qx', 'Qz', 't'


class QyQzIndexProjection(QxQyQzProjection):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        qx, qy, qz = super(QyQzIndexProjection, self).project(index, pdataframe)
        return qy, qz, numpy.ones_like(qy) * pdataframe.timestamp

    def get_axis_labels(self) -> Tuple[str]:
        return 'Qy', 'Qz', 't'


class QparQperProjection(QxQyQzProjection):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        qx, qy, qz = super(QparQperProjection, self).project(index, pdataframe)
        return numpy.sqrt(qx*qx + qy*qy), qz

    def get_axis_labels(self) -> Tuple[str]:
        return 'Qpar', 'Qper'


class QparQperIndexProjection(QparQperProjection):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        qpar, qper = super(QparQperIndexProjection, self).project(index, pdataframe)
        return qpar, qper, numpy.ones_like(qpar) * pdataframe.timestamp

    def get_axis_labels(self) -> Tuple[str]:
        return 'Qpar', 'Qper', 't'


class Stereo(QxQyQzProjection):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        qx, qy, qz = super(Stereo, self).project(index, pdataframe)
        q = numpy.sqrt(qx*qx+qy*qy+qz*qz)
        # ratio = qz + q
        # x = qx / ratio
        # y = qy / ratio
        return q, qx, qy

    def get_axis_labels(self) -> Tuple[str]:
        return "Q", "Qx", "Qy"

class QPolarProjection(QxQyQzProjection):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        qx, qy, qz = super(QPolarProjection, self).project(index, pdataframe)
        phi = numpy.rad2deg(numpy.arctan2(qx, qy))
        q = numpy.sqrt(qx*qx+qy*qy+qz*qz)
        return phi, q, qz

    def get_axis_labels(self) -> Tuple[str]:
        return "Phi", "Q", "Qz"

class QIndex(Stereo):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        q, qx, qy = super(QIndex, self).project(index, pdataframe)
        return q, numpy.ones_like(q) * pdataframe.timestamp

    def get_axis_labels(self) -> Tuple[str]:
        return "Q", "Index"


class AnglesProjection(backend.ProjectionBase):
    def project(self, index: int, pdataframe: PDataFrame) -> Tuple[ndarray]:
        # put the detector at the right position

        pixels, k, UB, R, P, idx, timestamp, _surface_orientation = pdataframe

        # on calcule le vecteur de l'axes de rotation de l'angle qui
        # nous interesse. (ici delta et gamma). example delta (0, 1,
        # 0) (dans le repere du detecteur). Il faut donc calculer la
        # matrice de transformation pour un axe donnée. C'est la liste
        # de transformations qui sont entre cet axe et le detecteur.
        axis_delta = None
        axis_gamma = None

        # il nous faut ensuite calculer la normale du plan dans lequel
        # nous allons projeter les pixels. (C'est le produit vectoriel
        # de k0, axis_xxx).
        n_delta = None
        n_gamma = None

        # On calcule la projection sur la normale des plans en
        # question.
        p_delta = None
        p_gamma = None

        # On calcule la norme de chaque pixel. (qui pourra etre
        # calcule une seule fois pour toutes les images).
        l2 = numpy.linalg.norm(pixels, order=2, axis=-1)

        # xxx0 is the angles of the diffractometer for the given
        # image.
        delta = numpy.arcsin(p_delta / l2) + delta0
        gamma = numpy.arcsin(p_gamma / l2) + gamma0
        omega = numpy.ones_like(delta) * omega0

        return (omega, delta, gamma)

    def get_axis_labels(self) -> Tuple[str]:
        return 'omega', 'delta', 'gamma'

###################
# Common methodes #
###################

WRONG_ATTENUATION = -100


def get_nxclass(hfile, nxclass, path="/"):
    """
    :param hfile: the hdf5 file.
    :type hfile: tables.file.
    :param nxclass: the nxclass to extract
    :type nxclass: str
    """
    for node in hfile.walk_nodes(path):
        try:
            if nxclass == as_string(node._v_attrs['NX_class']):
                return node
        except KeyError:
            pass
    return None


def node_as_string(node):
    if node.shape == ():
        content = node.read().tostring()
    else:
        content = node[0]
    return as_string(content)


Diffractometer = NamedTuple('Diffractometer',
                            [('name', str),  # name of the hkl diffractometer
                             ('ub', ndarray),  # the UB matrix
                             ('geometry', Hkl.Geometry)])  # the HklGeometry


def get_diffractometer(hfile):
    """ Construct a Diffractometer from a NeXus file """
    node = get_nxclass(hfile, 'NXdiffractometer')

    name = node_as_string(node.type)
    if name.endswith('\n'):
        # remove the last "\n" char
        name = name[:-1]

    try:
        ub = node.UB[:]
    except NoSuchNodeError:
        ub = None

    factory = Hkl.factories()[name]
    geometry = factory.create_new_geometry()

    # wavelength = get_nxclass(hfile, 'NXmonochromator').wavelength[0]
    # geometry.wavelength_set(wavelength)

    return Diffractometer(name, ub, geometry)


Sample = NamedTuple("Sample", [("a", float),
                               ("b", float),
                               ("c", float),
                               ("alpha", float),
                               ("beta", float),
                               ("gamma", float),
                               ("ux", float),
                               ("uy", float),
                               ("uz", float),
                               ("sample", Hkl.Sample)])


def get_sample(hfile):
    # hkl sample
    a = b = c = 1.54
    alpha = beta = gamma = 90
    ux = uy = uz = 0

    sample = Hkl.Sample.new("test")
    lattice = Hkl.Lattice.new(a, b, c,
                              math.radians(alpha),
                              math.radians(beta),
                              math.radians(gamma))
    sample.lattice_set(lattice)

    parameter = sample.ux_get()
    parameter.value_set(ux, Hkl.UnitEnum.USER)
    sample.ux_set(parameter)

    parameter = sample.uy_get()
    parameter.value_set(uy, Hkl.UnitEnum.USER)
    sample.uy_set(parameter)

    parameter = sample.uz_get()
    parameter.value_set(uz, Hkl.UnitEnum.USER)
    sample.uz_set(parameter)

    return Sample(1.54, 1.54, 1.54, 90, 90, 90, 0, 0, 0, sample)


Detector = NamedTuple("Detector", [("name", str),
                                   ("detector", Hkl.Detector)])


def get_detector(hfile, h5_nodes):
    detector = Hkl.Detector.factory_new(Hkl.DetectorType(0))
    images = h5_nodes["image"]
    s = images.shape[-2:]
    if s == (960, 560) or s == (560, 960):
        det = Detector("xpad_flat", detector)
    else:
        det = Detector("imxpads140", detector)
    return det


Source = NamedTuple("Source", [("wavelength", float)])


def get_source(hfile):
    wavelength = None
    node = get_nxclass(hfile, 'NXmonochromator')
    for attr in ["wavelength", "lambda"]:
        try:
            wavelength = node[attr][0]
        except NoSuchNodeError:
            pass
        except IndexError:
            pass

    return Source(wavelength)


DataFrame = NamedTuple("DataFrame", [("diffractometer", Diffractometer),
                                     ("sample", Sample),
                                     ("detector", Detector),
                                     ("source", Source),
                                     ("h5_nodes", Dict[str, Node])])


def dataframes(hfile, data_path=None):
    for group in hfile.get_node('/'):
        scan_data = group._f_get_child("scan_data")
        # now instantiate the pytables objects
        h5_nodes = {}
        for key, hitem in data_path.items():
            try:
                child = scan_data._f_get_child(hitem.name)
            except tables.exceptions.NoSuchNodeError:
                if hitem.optional:
                    child = None
                else:
                    raise
            h5_nodes[key] = child

    diffractometer = get_diffractometer(hfile)
    sample = get_sample(hfile)
    detector = get_detector(hfile, h5_nodes)
    source = get_source(hfile)

    yield DataFrame(diffractometer, sample, detector, source, h5_nodes)


def get_ki(wavelength):
    """
    for now the direction is always along x
    """
    TAU = 2 * math.pi
    return numpy.array([TAU / wavelength, 0, 0])


def normalized(a, axis=-1, order=2):
    l2 = numpy.atleast_1d(numpy.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / numpy.expand_dims(l2, axis)


def hkl_matrix_to_numpy(m):
    M = numpy.empty((3, 3))
    for i in range(3):
        for j in range(3):
            M[i, j] = m.get(i, j)
    return M


def M(theta, u):
    """
    :param theta: the axis value in radian
    :type theta: float
    :param u: the axis vector [x, y, z]
    :type u: [float, float, float]
    :return: the rotation matrix
    :rtype: numpy.ndarray (3, 3)
    """
    c = cos(theta)
    one_minus_c = 1 - c
    s = sin(theta)
    return numpy.array([[c + u[0]**2 * one_minus_c,
                         u[0] * u[1] * one_minus_c - u[2] * s,
                         u[0] * u[2] * one_minus_c + u[1] * s],
                        [u[0] * u[1] * one_minus_c + u[2] * s,
                         c + u[1]**2 * one_minus_c,
                         u[1] * u[2] * one_minus_c - u[0] * s],
                        [u[0] * u[2] * one_minus_c - u[1] * s,
                         u[1] * u[2] * one_minus_c + u[0] * s,
                         c + u[2]**2 * one_minus_c]])


##################
# Input Backends #
##################

class SIXS(backend.InputBase):
    # OFFICIAL API

    dbg_scanno = None
    dbg_pointno = None

    def generate_jobs(self, command):
        scans = util.parse_multi_range(','.join(command).replace(' ', ','))
        if not len(scans):
            sys.stderr.write('error: no scans selected, nothing to do\n')
        for scanno in scans:
            util.status('processing scan {0}...'.format(scanno))
            if self.config.pr:
                pointcount = self.config.pr[1] - self.config.pr[0] + 1
                start = self.config.pr[0]
            else:
                start = 0
                pointcount = self.get_pointcount(scanno)
            if pointcount > self.config.target_weight * 1.4:
                for s in util.chunk_slicer(pointcount,
                                           self.config.target_weight):
                    yield backend.Job(scan=scanno,
                                      firstpoint=start+s.start,
                                      lastpoint=start+s.stop-1,
                                      weight=s.stop-s.start)
            else:
                yield backend.Job(scan=scanno,
                                  firstpoint=start,
                                  lastpoint=start+pointcount-1,
                                  weight=pointcount)

    def process_job(self, job):
        super(SIXS, self).process_job(job)
        with tables.open_file(self.get_filename(job.scan), 'r') as scan:
            self.metadict = dict()
            try:
                for dataframe in dataframes(scan, self.HPATH):
                    pixels = self.get_pixels(dataframe.detector)
                    detector = ALL_DETECTORS[dataframe.detector.name]()
                    mask = detector.mask.astype(numpy.bool)
                    maskmatrix = load_matrix(self.config.maskmatrix)
                    if maskmatrix is not None:
                        mask = numpy.bitwise_or(mask, maskmatrix)

                    for index in range(job.firstpoint, job.lastpoint + 1):
                        yield self.process_image(index, dataframe, pixels, mask)  # noqa
                util.statuseol()
            except Exception as exc:
                exc.args = errors.addmessage(exc.args, ', An error occured for scan {0} at point {1}. See above for more information'.format(self.dbg_scanno, self.dbg_pointno))  # noqa
                raise
            self.metadata.add_section('sixs_backend', self.metadict)

    def parse_config(self, config):
        super(SIXS, self).parse_config(config)
        # Optional, select a subset of the image range in the x
        # direction. all by default
        self.config.xmask = util.parse_multi_range(config.pop('xmask', None))

        # Optional, select a subset of the image range in the y
        # direction. all by default
        self.config.ymask = util.parse_multi_range(config.pop('ymask', None))

        # location of the nexus files (take precedence on nexusfile)
        self.config.nexusdir = config.pop('nexusdir', None)

        # Location of the specfile
        self.config.nexusfile = config.pop('nexusfile', None)

        # Optional, all range by default
        self.config.pr = config.pop('pr', None)
        if self.config.xmask is None:
            self.config.xmask = slice(None)
        if self.config.ymask is None:
            self.config.ymask = slice(None)
        if self.config.pr:
            self.config.pr = util.parse_tuple(self.config.pr, length=2, type=int)  # noqa

        # sample to detector distance (mm)
        self.config.sdd = float(config.pop('sdd'))

        # x,y coordinates of the central pixel
        self.config.centralpixel = util.parse_tuple(config.pop('centralpixel'), length=2, type=int)  # noqa

        # Optional, if supplied pixels where the mask is 0 will be removed
        self.config.maskmatrix = config.pop('maskmatrix', None)

        # detector rotation around x (1, 0, 0)
        self.config.detrot = config.pop('detrot', None)
        if self.config.detrot is not None:
            try:
                self.config.detrot = float(self.config.detrot)
            except ValueError:
                self.config.detrot = None

        # attenuation_coefficient (Optional)
        attenuation_coefficient = config.pop('attenuation_coefficient', None)
        if attenuation_coefficient is not None:
            try:
                self.config.attenuation_coefficient = float(attenuation_coefficient)  # noqa
            except ValueError:
                self.config.attenuation_coefficient = None
        else:
            self.config.attenuation_coefficient = None

        # surface_orientation
        surface_orientation = config.pop('surface_orientation', None)
        surface_orientation_opt = SurfaceOrientation.VERTICAL
        if surface_orientation is not None:
            if surface_orientation.lower() == "horizontal":
                surface_orientation_opt = SurfaceOrientation.HORIZONTAL
        self.config.surface_orientation = surface_orientation_opt

    def get_destination_options(self, command):
        if not command:
            return False
        command = ','.join(command).replace(' ', ',')
        scans = util.parse_multi_range(command)
        return dict(first=min(scans), last=max(scans), range=','.join(str(scan) for scan in scans))  # noqa

    # CONVENIENCE FUNCTIONS
    def get_filename(self, scanno):
        filename = None
        if self.config.nexusdir:
            dirname = self.config.nexusdir
            files = [f for f in os.listdir(dirname)
                     if ((str(scanno).zfill(5) in f)
                         and (os.path.splitext(f)[1] in ['.hdf5', '.nxs']))
                     ]
            if files is not []:
                filename = os.path.join(dirname, files[0])
        else:
            filename = self.config.nexusfile.format(scanno=str(scanno).zfill(5))  # noqa
        if not os.path.exists(filename):
            raise errors.ConfigError('nexus filename does not exist: {0}'.format(filename))  # noqa
        return filename

    @staticmethod
    def apply_mask(data, xmask, ymask):
        roi = data[ymask, :]
        return roi[:, xmask]


HItem = namedtuple("HItem", ["name", "optional"])


class FlyScanUHV(SIXS):
    HPATH = {
        "image": HItem("xpad_image", False),
        "mu": HItem("UHV_MU", False),
        "omega": HItem("UHV_OMEGA", False),
        "delta": HItem("UHV_DELTA", False),
        "gamma": HItem("UHV_GAMMA", False),
        "attenuation": HItem("attenuation", True),
        "timestamp": HItem("epoch", True),
    }

    def get_pointcount(self, scanno):
        # just open the file in order to extract the number of step
        with tables.open_file(self.get_filename(scanno), 'r') as scan:
            return get_nxclass(scan, "NXdata").xpad_image.shape[0]

    def get_attenuation(self, index, h5_nodes, offset):
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
            except IndexError:
                attenuation = WRONG_ATTENUATION
        return attenuation

    def get_timestamp(self, index, h5_nodes):
        timestamp = None
        try:
            timestamp = h5_nodes['timestamp'][index]
        except KeyError:
            timestamp = index
        return timestamp

    def get_values(self, index, h5_nodes):
        image = h5_nodes['image'][index]
        mu = h5_nodes['mu'][index]
        omega = h5_nodes['omega'][index]
        delta = h5_nodes['delta'][index]
        gamma = h5_nodes['gamma'][index]
        attenuation = self.get_attenuation(index, h5_nodes, 2)
        timestamp = self.get_timestamp(index, h5_nodes)

        return (image, attenuation, timestamp, (mu, omega, delta, gamma))

    def process_image(self, index, dataframe, pixels, mask):
        util.status(str(index))

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


class FlyScanUHV2(FlyScanUHV):
    HPATH = {
        "image": HItem("xpad_image", False),
        "mu": HItem("mu", False),
        "omega": HItem("omega", False),
        "delta": HItem("delta", False),
        "gamma": HItem("gamma", False),
        "attenuation": HItem("attenuation", True),
        "timestamp": HItem("epoch", True),
    }


class FlyMedH(FlyScanUHV):
    HPATH = {
        "image": HItem("xpad_image", False),
        "pitch": HItem("beta", True),
        "mu": HItem("mu", False),
        "gamma": HItem("gamma", False),
        "delta": HItem("delta", False),
        "attenuation": HItem("attenuation", True),
        "timestamp": HItem("epoch", True),
    }

    def get_values(self, index, h5_nodes):
        image = h5_nodes['image'][index]
        pitch = h5_nodes['pitch'][index] if h5_nodes['pitch'] else 0.3
        mu = h5_nodes['mu'][index]
        gamma = h5_nodes['gamma'][index]
        delta = h5_nodes['delta'][index]
        attenuation = self.get_attenuation(index, h5_nodes, 2)
        timestamp = self.get_timestamp(index, h5_nodes)

        return (image, attenuation, timestamp, (pitch, mu, gamma, delta))


class SBSMedH(FlyScanUHV):
    HPATH = {
        "image": HItem("data_03", False),
        "pitch": HItem("data_22", False),
        "mu": HItem("data_18", False),
        "gamma": HItem("data_20", False),
        "delta": HItem("data_19", False),
        "attenuation": HItem("data_xx", True),
        "timestamp": HItem("epoch", True),
    }

    def get_pointcount(self, scanno):
        # just open the file in order to extract the number of step
        with tables.open_file(self.get_filename(scanno), 'r') as scan:
            return get_nxclass(scan, "NXdata").data_03.shape[0]

    def get_values(self, index, h5_nodes):
        image = h5_nodes['image'][index]
        pitch = h5_nodes['pitch'][index]
        mu = h5_nodes['mu'][index]
        gamma = h5_nodes['gamma'][index]
        delta = h5_nodes['delta'][index]
        attenuation = self.get_attenuation(index, h5_nodes, 2)
        timestamp = self.get_timestamp(index, h5_nodes)

        return (image, attenuation, timestamp, (pitch, mu, gamma, delta))


class SBSFixedDetector(FlyScanUHV):
    HPATH = {
        "image": HItem("data_11", False),
        "timestamp": HItem("sensors_timestamps", True),
    }

    def get_pointcount(self, scanno):
        # just open the file in order to extract the number of step
        with tables.open_file(self.get_filename(scanno), 'r') as scan:
            return get_nxclass(scan, "NXdata").data_11.shape[0]

    def get_values(self, index, h5_nodes):
        image = h5_nodes['image'][index]
        attenuation = self.get_attenuation(index, h5_nodes, 2)
        timestamp = self.get_timestamp(index, h5_nodes)
        print(timestamp, type(timestamp))

        return (image, attenuation, timestamp, None)

    def process_image(self, index, dataframe, pixels, mask):
        util.status(str(index))

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

        I = numpy.array([[1,  0, 0],
                         [0,  0, 1],
                         [0, -1, 0]])


        # hkl_geometry = dataframe.diffractometer.geometry
        # hkl_geometry.axis_values_set(values, Hkl.UnitEnum.USER)

        # # sample
        # hkl_sample = dataframe.sample.sample
        # q_sample = hkl_geometry.sample_rotation_get(hkl_sample)
        # R = hkl_matrix_to_numpy(q_sample.to_matrix())

        # # detector
        # hkl_detector = dataframe.detector.detector
        # q_detector = hkl_geometry.detector_rotation_get(hkl_detector)
        # P = hkl_matrix_to_numpy(q_detector.to_matrix())

        if self.config.detrot is not None:
            P = M(math.radians(self.config.detrot), [1, 0, 0])

        pdataframe = PDataFrame(pixels, k, I, I, P, index, timestamp)

        return intensity, weights, (index, pdataframe)



class FlyMedV(FlyScanUHV):
    HPATH = {
        "image": HItem("xpad_image", False),
        "beta": HItem("beta", True),
        "mu": HItem("mu", False),
        "omega": HItem("omega", False),
        "gamma": HItem("gamma", False),
        "delta": HItem("delta", False),
        "etaa": HItem("etaa", True),
        "attenuation": HItem("attenuation", True),
        "timestamp": HItem("epoch", True),
    }

    def get_values(self, index, h5_nodes):
        image = h5_nodes['image'][index]
        beta = h5_nodes['beta'][index] if h5_nodes['beta'] else 0.0
        mu = h5_nodes['mu'][index]
        omega = h5_nodes['omega'][index]
        gamma = h5_nodes['gamma'][index]
        delta = h5_nodes['delta'][index]
        etaa = h5_nodes['etaa'][index] if h5_nodes['etaa'] else 0.0
        attenuation = self.get_attenuation(index, h5_nodes, 2)
        timestamp = self.get_timestamp(index, h5_nodes)

        return (image, attenuation, timestamp, (beta, mu, omega, gamma, delta, etaa))


def load_matrix(filename):
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
