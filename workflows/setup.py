import os

from setuptools import find_packages, setup


def read_requirements(pkg_name):
    """Read requirements from requirements.txt if it exists"""
    if os.path.exists(f'{pkg_name}/requirements.txt'):
        with open(f'{pkg_name}/requirements.txt', 'r', encoding="utf-8") as req:
            return req.read().splitlines()
    return []


setup(
    name='robotic_ultrasound',
    version='0.0.1',
    package_dir={'robotic_ultrasound': 'robotic_ultrasound'},
    packages=find_packages(),
    install_requires=read_requirements('robotic_ultrasound'),
)
