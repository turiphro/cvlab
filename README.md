This repository contains code and resources for
my personal computer vision lab. It will be a
playground for new ideas using a diversity of
(Python and C++) libraries, 2D and 3D input
devices (such as webcams, Kinect, Leap Motion 
and Wiimotes), prototypes, and documentation.

As this repo is meant for experimentation and
prototyping, please do not judge code quality ;)


Installation (Ubuntu)
=====================

# General
# Notes:
# - pip's pil is Pillow, PIL's fork
# - pip's matplotlib includes pylab; it needs cairocffi, which needs libffi-dev.

sudo apt-get install python-pip python3-pip python3-numpy python3-dev potrace libgtk2.0-dev libeigen3-dev libavcodec-dev libdc1394-22-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libopenni-dev imagemagick libffi-dev
sudo pip3 install pillow numpy scipy matplotlib cairocffi

# OpenCV
Compile OpenCV 4.0+, including Python (3) bindings

https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/


Tools
=====

# Interactive terminal
ipython3
import pylab as plt
import numpy as np
import scipy as sp

# For interactive notebook, use:
ipython3 notebook   # opens in browser
%pylab inline


Documentation
=============

For more notes, see the doc/ directory.
