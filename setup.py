import setuptools
from setuptools import find_packages, setup

if __name__ == "__main__":
    setuptools.setup(
        name="mxblas",
        version="1.0.0",
        packages=find_packages(),
        cmdclass={},
    )
