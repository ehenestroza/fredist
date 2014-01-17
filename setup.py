#!/usr/bin/env python

from distutils.core import setup

setup(name='fredist',
      version='2.0',
      author='Enrique Henestroza Anguiano',
      author_email='ehenestroza@gmail.com',
      url='http://alpage.inria.fr/~henestro',
      description='Dependency Parsing and Distributional Lexical Methods',
      license='GPL',
      packages=['fredist'],
      package_dir={'fredist': 'src/fredist'},
      )
