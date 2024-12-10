#!/bin/bash

# Set the project name
PROJECT_NAME="measurement-processor"

# Create the main project directory
mkdir $PROJECT_NAME

# Navigate into the project directory
cd $PROJECT_NAME

# Create the main package directory
mkdir measurement_processor

# Create __init__.py to make it a package
touch measurement_processor/__init__.py

# Create modules for the package
touch measurement_processor/data_handler.py
touch measurement_processor/preprocessor.py
touch measurement_processor/feature_extractor.py
touch measurement_processor/visualizer.py
touch measurement_processor/model_interface.py
touch measurement_processor/data_exporter.py
touch measurement_processor/config_manager.py

# Create a directory for tests
mkdir tests
touch tests/test_example.py

# Create the setup.py file
cat <<EOL > setup.py
from setuptools import setup, find_packages

setup(
    name="$PROJECT_NAME",
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
EOL

# Create requirements.txt
cat <<EOL > requirements.txt
pandas
numpy
matplotlib
scikit-learn
EOL

# Create a .gitignore file
cat <<EOL > .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
*.coverage
.cache
nosetests.xml
coverage.xml
*.log

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# mypy
.mypy_cache/
.dmypy.json
dmypy.json
EOL

# Create a LICENSE file (MIT License example)
cat <<EOL > LICENSE
MIT License

Copyright (c) $(date +%Y) Jan Büscher

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOL

echo "Project structure created successfully."
