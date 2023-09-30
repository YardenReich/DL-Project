import os

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name="vqgan",
    py_modules=["VQGAN"],
    version="1.0",
    description="",
    author="Shahar, Yarden",
    install_requires=install_requires,
)