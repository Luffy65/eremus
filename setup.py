#! /usr/bin/env python
#

from setuptools import setup, find_packages

setup(
    name='eremus',
    url='https://github.com/SalvoCalcagno/eremus',
    author='Salvatore Calcagno',
    author_email='salvo.calcagno@hotmail.it',
    packages=find_packages(),#['eremus, eremus.hello'],
    install_requires=['numpy'],
    version='0.1',
    license='MIT',
    description='Utilities for EREMUS dataset',
    include_package_data=True,
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
