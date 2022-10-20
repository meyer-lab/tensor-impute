try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='tensor-impute',
      version='0.1.0',
      description='Examining various imputation methods for tensor decomposition.',
      url='https://github.com/meyer-lab/tensorpack',
      license='MIT',
      packages=find_packages(exclude=['doc']),
      install_requires=['numpy', 'tensorly', 'scikit-learn'])
