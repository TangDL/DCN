#!/bin/bash

mkdir -p ./data
mkdir -p ./output
mkdir -p ./external/mxnet
mkdir -p ./model/pretrained_model

cd lib/bbox
/root/software/anaconda2/bin/python setup_linux.py build_ext --inplace
cd ../dataset/pycocotools
/root/software/anaconda2/bin/python setup_linux.py build_ext --inplace
cd ../../nms
/root/software/anaconda2/bin/python setup_linux.py build_ext --inplace
cd ../..
set ff=unix
