#!/bin/bash

apt-get install -y wget bash git gcc g++ gfortran  liblapack-dev libamd2 libcholmod3 libmetis-dev libsuitesparse-dev libnauty2-dev
cd ~/
wget -nH https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
chmod u+x coinbrew
bash coinbrew fetch Cbc@master
bash coinbrew build Cbc@master --no-prompt --prefix=/usr/local --tests=none --enable-cbc-parallel
sudo apt-get install -y cmake
sudo apt-get install -y libblas-dev liblapack-dev
