#!/bin/bash
#python cnn_sat.py
convert_imageset data/ data/train_listfile data/train_lmdb
convert_imageset data/ data/val_listfile data/val_lmdb
compute_image_mean data/train_lmdb data/train_mean_image -backend "lmdb"
#wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel net1/
