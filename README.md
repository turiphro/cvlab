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

Last tested on Ubuntu 19.10.

```
sudo apt-get install python-pip python3-pip python3-numpy python3-dev potrace libgtk2.0-dev libeigen3-dev libavcodec-dev libdc1394-22-dev libjpeg-dev libpng-dev libtiff-dev libopenni-dev imagemagick libffi-dev

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


Or install manually:

## OpenCV (manual installation)
Compile OpenCV 4.0+, including Python (3) bindings (6min on my core i7)
https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/

    export OPENCV=4.1.2  # 2019-10-10
    wget -O opencv.zip https://github.com/opencv/opencv/archive/$OPENCV.zip
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/$OPENCV.zip
    unzip opencv.zip && unzip opencv_contrib.zip
    mv opencv-$OPENCV opencv && mv opencv_contrib-$OPENCV opencv_contrib

    mkdir -p opencv/build/ && cd opencv/build
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_EXTRA_MODULES_PATH=$PWD/../../opencv_contrib/modules \
	-D PYTHON_EXECUTABLE=$(which python3) \
	-D PYTHON2_EXECUTABLE=$(which python2) \
	-D BUILD_EXAMPLES=ON ..
    make -j5
    sudo make install


Tools
=====

## Inference on webcams
We can run various inference algorithms on streams of images (e.g., webcams or folders with images).

Example:

```
export PYTHONPATH="$PWD:$PYTHONPATH"
python3 apps/viewer.py -i 0 1 2 -f classic.OpticalFlow
```

You can find all available algorithms with:

``` grep "(\w+)\(Inference\)" -Eo inference -r ```


For more notes, see the doc/ directory.
