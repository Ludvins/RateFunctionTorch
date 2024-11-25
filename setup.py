from setuptools import setup, find_packages

setup(
    name="ratefunctiontorch",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm"
    ]
) 