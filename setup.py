from setuptools import find_namespace_packages
from setuptools import setup

requirements = open('requirements.txt').readlines()

packages = find_namespace_packages()

packages = [f for f in packages if "dragtraffic" in f]

setup(name='dragtraffic', packages=packages, install_requires=requirements)
