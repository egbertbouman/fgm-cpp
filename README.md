# fgm-cpp
C++ library for solving graph matching problems using Factorized Graph Matching

## Build instructions for Ubuntu
You will need the GNU G++ compiler and cmake:

```apt-get install build-essential cmake```

Make sure you have libboost-python, numpy, and eigen3 installed:

```sudo apt-get install libboost-python1.55 python-numpy libeigen3-dev```

Next, you can generate the makefile using cmake and build fgm-cpp:

```
cd ~/git/fgm-cpp
cmake .
make
```

If you get the error like ```fatal error: Eigen/Dense: No such file or directory```, please try the following:

```
cd /usr/include
sudo ln -sf eigen3/Eigen Eigen
```
