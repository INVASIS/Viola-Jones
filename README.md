/!\ WORK IN PROGRESS /!\

# Viola Jones implementation

Based on the following paper: http://www.ipol.im/pub/art/2014/104/article.pdf

[![Build Status][travis-image]][travis-url] [![License][license-image]][license-url]

### Specifications

Language: Java (JDK 8)

### Compatibility

Tested for:
* Windows 10 X64
* Ubuntu 16.04 X64

### Included dependencies

* [JCuda](http://www.jcuda.org/): Java bindings for CUDA (only used if available)
* [Jeigen](https://github.com/hughperkins/jeigen): Java wrapper for Eigen C++ fast matrix library

### Optional requirements

* [CUDA Toolkit 7.5](https://developer.nvidia.com/cuda-toolkit) 

## Installation

We provide .dll (for windows) and .so (for linux) CUDA 7.5 lib files in this repository.
In order to let the program access to these files, you must add their directory in your PATH environment variable.


## References

1. [An Analysis of the Viola-Jones Face Detection Algorithm](http://www.ipol.im/pub/art/2014/104/article.pdf)
2. [Robust Real-Time Face Detection](http://www.face-rec.org/algorithms/Boosting-Ensemble/16981346.pdf)
3. [Adam Harvey Explains Viola-Jones Face Detection](http://www.makematics.com/research/viola-jones/)
4. [5KK73 GPU Assignment 2012](https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection)
5. [Object detection using cascades of boosted classifiers](http://www.die.uchile.cl/ieee-cis/files/RuizdelSolar_T9.pdf)

[travis-url]: https://travis-ci.org/INVASIS/Viola-Jones
[travis-image]: http://img.shields.io/travis/INVASIS/Viola-Jones.svg?style=flat-square
[license-image]: http://img.shields.io/badge/license-MIT-green.svg?style=flat-square
[license-url]: LICENSE
