from setuptools import setup, find_packages

setup(
    name="measurement-processor",
    version="0.1.0",
    description="A Python package for processing data from various measurement systems.",
    author="Your Name",
    author_email="your_email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
    ],
)
