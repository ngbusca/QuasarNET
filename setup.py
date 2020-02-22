#!/usr/bin/env python

import glob

from setuptools import setup, find_packages

scripts = glob.glob('bin/*')

description = "CNN for quasar classification and redshifting"

version="2.0"
setup(name="quasarnet",
      version=version,
      description=description,
      url="https://github.com/ngbusca/QuasarNET",
      author="Nicolas Busca et al",
      author_email="nbusca@gmail.com",
      packages=['quasarnet'],
      package_dir = {'': 'py'},
      install_requires=['scipy','numpy',
          'fitsio','h5py','tensorflow'],
      #test_suite='picca.test.test_cor',
      scripts = scripts
      )

