#!/usr/bin/env python

from distutils.core import setup, Extension

setup(
    name = 'Python Deep Microscopy toolbox',
    version = '0.1',
    author = 'Philipp Pelz',
    description = 'Deep reconstruction toolbox', 
    long_description = file('README.md','r').read(),
    package_dir = {'dptycho':'dptycho'},
    packages = ['dptycho',
                'dptycho.core',\
                'dptycho.utils',\
                'dptycho.simulations',\
                'dptycho.io',\
                'dptycho.experiment']
    )
