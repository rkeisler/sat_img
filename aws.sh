cd /root/
sudo apt-get update
# you'll need to put in "Y" during the following installations.
sudo apt-get install libatlas-base-dev
sudo apt-get install libxi-dev libxmu-dev freeglut3-dev build-essential binutils-gold
sudo apt-get remove libopenblas-base

pip install --upgrade numpy
pip install ipdb

#edit .bashrc:
#alias lsl='ls -lrth'
#alias e='emacs -nw'

#e .emacs.d/init.el
#(global-set-key [(control h)] 'delete-backward-char)

##########################################################
# REMAINING
##########################################################
git clone https://github.com/rkeisler/cuda-convnet
mkdir /root/cuda-convnet/dummyinclude/
mv cutil_inline.h /root/cuda-convnet/dummyinclude/
mv common-gcc-cuda-4.0.mk /root/cuda-convnet/
mv Makefile /root/cuda-convnet/
mv build.sh /root/cuda-convnet/


wget http://developer.download.nvidia.com/compute/cuda/5_0/rel-update-1/installers/cuda_5.0.35_linux_64_ubuntu11.10-1.run
/sbin/init 3
sudo ln -s /usr/lib/x86_64-linux-gnu/libglut.so.3 /usr/lib/libglut.so
# only install the Samples (option 3)
# accept, n, n, y, [enter], [enter]
sudo sh cuda_5.0.35_linux_64_ubuntu11.10-1.run


cd /root/cuda-convnet
sudo sh build.sh

## uncomment this if you want to try to train on CIFAR.
##sh /root/test_cifar.sh


