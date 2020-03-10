# Quantized Versions of VGG on CIFAR-10  

This repository contains the code producing the figures of the paper Low Power In-Memory Implementation of  Ternary Neural Networks  with Resistive RAM-Based Synapse. To set the environment run in your conda main environment:  
> conda config --add channels conda-forge  
> conda create --name environment_name --file requirements.txt  
> conda activate environment_name  
> conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch  

The code for BNN modules was adapted from https://github.com/itayhubara/BinaryNet.pytorch  
The code for VGG architecture was adapted from https://github.com/kuangliu/pytorch-cifar (Copyright (c) 2017 liukuang)  

## Training Quantized VGGs  

> python main.py --filt 128 --weight-quant T --act-quant F --ber 0.0 --optim adamw --decay 2.0 --lr 0.01 --mbs 128 --epochs 700 --device 0 --save True

