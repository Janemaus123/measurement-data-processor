from setuptools import setup, find_packages

setup(
    name="measurement-processor",
    version="0.1.0",
    description="A Python package for processing data from various measurement systems.",
    author="Jan Buescher",
    author_email="jan.buescher@tu-odrtmund.de",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
    ],
)
