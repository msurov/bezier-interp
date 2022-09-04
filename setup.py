#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
      name='bezier-interp',
      version='0.1',
      python_requires=">=3.6",
      description='Python Bezier Curves',
      author='Maksim Surov',
      author_email='surov.m.o@mail.com',
      packages=find_packages(where="src"),
      package_dir={"": "src"},
      install_requires=['numpy', 'scipy'],
)
