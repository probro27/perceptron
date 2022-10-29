from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'A basic Perceptron Linear Classifier algorithm'
LONG_DESCRIPTION = 'A package that allows to build a simple linear classifier Perceptron algorithm provided the data and the labels.'

# Setting up
setup(
    name="perceptron-linear-classifier",
    version=VERSION,
    author="Prabhav Khera",
    author_email="prabhavkhera@hotmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib'],
    test_requires=['pytest'],
    keywords=['python', 'perceptron', 'linear classifier', 'classifier'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
