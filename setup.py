#!/usr/bin/env python

from distutils.core import setup

setup(
      name='bezier-curves',
      version='0.1',
      description='Python Bezier Curves',
      author='Maksim Surov',
      author_email='surov.m.o@mail.com',
      packages=['.'],
      install_requires=['numpy', 'scipy'],
)
