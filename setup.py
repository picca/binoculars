import os

from setuptools import setup, find_packages

description = ("Data reduction and analysis software for two-dimensional "
               "detectors in surface X-ray diffraction")

long_description =\
 """ BINoculars is a tool for data reduction and analysis of large sets of
 surface diffraction data that have been acquired with a
 two-dimensional X-ray detector. The intensity of each pixel of a
 two-dimensional detector is projected onto a three-dimensional grid
 in reciprocal-lattice coordinates using a binning algorithm. This
 allows for fast acquisition and processing of high-resolution data
 sets and results in a significant reduction of the size of the data
 set. The subsequent analysis then proceeds in reciprocal space. It
 has evolved from the specific needs of the ID03 beamline at the ESRF,
 but it has a modular design and can be easily adjusted and extended
 to work with data from other beamlines or from other measurement
 techniques."""

scripts = [os.path.join("scripts", d)
           for d in ["binoculars-fitaid",
                     "binoculars-gui",
                     "binoculars-processgui",
                     "binoculars"]]

install_requires = ['h5py',
                    'numpy',
                    'matplotlib',
                    'pyFAI',
                    'PyMca5',
                    'PyQt5',
                    'vtk7']

setup(name='binoculars', version='0.0.5-dev',
      description=description,
      long_description=long_description,
      packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
      install_requires=install_requires,
      scripts=scripts,
      author="Willem Onderwaater, Sander Roobol, Frédéric-Emmanuel Picca",
      author_email="onderwaa@esrf.fr, picca@synchrotron-soleil.fr",
      url='FIXME',
      license='GPL-3',
      classifiers=[
          'Topic :: Scientific/Engineering',
          'Development Status :: 3 - Alpha',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Programming Language :: Python :: 3.7']
      )
