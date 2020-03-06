import numpy as np
import pandas as pd
import math
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import argparse

from model_utils import *
from collections import OrderedDict
from datetime import datetime
from PIL import Image


parser = argparse.ArgumentParser(description='Comparison Between Binary, Ternary, Float VGGs')

parser.add_argument('--lr', type = float, default = 0.0001, metavar = 'LR', help='Learning rate')
parser.add_argument('--mbs', type = int, default = 64, metavar = 'MBS', help='MiniBatch size')
parser.add_argument('--filt', type = int, default = 128, metavar = 'F', help='Number of filters in input layer')
parser.add_argument('--threshold', type = float, default = 0.05, metavar = 'Th', help='Threshold for ternary weight')
parser.add_argument('--optim', type = str, default = 'adam', metavar = 'opt', help='optimizer between sgd and adam')
parser.add_argument('--weight-quant', type = str, default = 'F', metavar = 'q', help='B, T, F')
parser.add_argument('--act-quant', type = str, default = 'F', metavar = 'q', help='B, T, F')
parser.add_argument('--epochs', type = int, default = 400, metavar = 'ep', help='Number of epochs')
parser.add_argument('--decay',type = float, default = 0.0, metavar = 'd', help='L2 regularization term')
parser.add_argument('--ber',type = float, default = 0.0, metavar = 'be', help='Bit error rate')
parser.add_argument('--momentum', type = float, default = 0.0, metavar = 'mntm', help='Momentum for sgd')
parser.add_argument('--save',type = bool, default = True, metavar = 'S', help='Saving the results')
parser.add_argument('--learned-bn',type = bool, default = True, metavar = 'LBN', help='Learning Batch Norm parameters')
parser.add_argument('--device',type = int, default = 0, metavar = 'Dev', help='choice of gpu')

args = parser.parse_args()

date = datetime.now().strftime('%Y-%m-%d')
time = datetime.now().strftime('%H-%M-%S')
path = 'results/'+date+'/'+time+'_gpu'+str(args.device)
if not(os.path.exists(path)):
    os.makedirs(path)

device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

createHyperparametersFile(path, args)

# Hyperparameters
epochs = args.epochs
save_result = args.save

filt = args.filt
#conv_list = [64, 'B', 64, 'B', 'M', 128, 'B', 128, 'B', 'M', 256, 'B', 256, 'B', 256, 'B', 'M', 512, 'B', 512, 'B', 512, 'B', 'M', 512, 'B', 512, 'B', 512, 'B', 'M']
conv_list = [filt, 'B', filt, 'M', 'B', 2*filt, 'B', 2*filt, 'M', 'B', 4*filt, 'B', 4*filt, 'M', 'B']
fc_list = [512]

model = VGG(conv_list, fc_list, args.weight_quant, args.act_quant, args.ber, aff=args.learned_bn, threshold=args.threshold).to(device)
model.apply(normal_init)

print(date+'_'+time)
print(model)
plot_parameters(model, path, save=save_result)


# Data preparation 
transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(0.5),
                                                  #torchvision.transforms.RandomChoice([torchvision.transforms.RandomRotation(10), torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge')]),
                                                  torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
                                                  torchvision.transforms.ToTensor(), 
                                                  torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                                                 ])   

transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )])
transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )]) 


cifar10_train_dset = torchvision.datasets.CIFAR10('./cifar10_pytorch', train=True, transform=transform_train, download=False)
cifar10_test_dset = torchvision.datasets.CIFAR10('./cifar10_pytorch', train=False, transform=transform_test, download=False)

print(cifar10_train_dset.transform)

# For Validation set
#val_index = np.random.randint(10)
#val_samples = list(range( 5000 * val_index, 5000 * (val_index + 1) ))
#train_samples = list(range(0, 5000 * val_index))   +   list(range(5000 * (val_index + 1), 50000))
#train_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=args.mbs, sampler = torch.utils.data.SubsetRandomSampler(train_samples), shuffle=False, num_workers=1)
#val_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=args.mbs, sampler = torch.utils.data.SubsetRandomSampler(val_samples), shuffle=False, num_workers=1)

train_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=args.mbs, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(cifar10_test_dset, batch_size=256, shuffle=False, num_workers=1)


# Result collect initialisation
data = {}
data['net'] = model.__class__.__name__
data['WQ'], data['AQ'] = args.weight_quant, args.act_quant
data['lr'], data['mbs'], data['epoch'], data['tr_loss'], data['acc_tr'], data['acc_test'] = [], [], [], [], [], []
#data['val_loss'], data['acc_val'] = [], []
name = data['net']+'_CIFAR10'


if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,250,350], gamma=0.1)
elif args.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,250,350], gamma=0.1)
elif args.optim == 'adamw':
    T_max = 100
    cumul_T_max = 100
    decay_W = args.decay*math.sqrt(args.mbs/(len(train_loader.dataset)*T_max))
    optimizer = AdamW(model.parameters(), lr = args.lr, weight_decay = decay_W)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)

print(optimizer)

for epoch in range(1, epochs+1):

    train(model, train_loader, optimizer, device)

    train_accuracy, train_loss = test(model, train_loader, device)
    #val_accuracy, val_loss = test(model, val_loader, device, frac = 10)
    test_accuracy, _ = test(model, test_loader, device)

    data['epoch'].append(epoch)
    data['lr'].append(optimizer.param_groups[0]['lr'])
    data['mbs'].append(train_loader.batch_size)
    data['acc_tr'].append(train_accuracy)
    #data['acc_val'].append(val_accuracy)
    data['acc_test'].append(test_accuracy)
    data['tr_loss'].append(train_loss)
    #data['val_loss'].append(val_loss)

    scheduler.step()

    if args.optim == 'adamw':
        if epoch==cumul_T_max:
            T_max *= 2
            cumul_T_max += T_max
            decay_W = args.decay*math.sqrt(args.mbs/(len(train_loader.dataset)*T_max))
            optimizer = AdamW(model.parameters(), lr = args.lr, weight_decay = decay_W)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)     

    if (epoch%10==0) and save_result:
        df_data = pd.DataFrame(data)
        df_data.to_csv(path +'/'+name+'.csv', index = False)
        plot_acc(path, name+'.csv')
        #plot_loss(path, name+'.csv')
    plot_parameters(model, path, save=save_result)


df_data = pd.DataFrame(data)
df_data.to_csv(path +'/'+name+'.csv', index = False)
plot_acc(path, name+'.csv')

if save_result:
    torch.save({'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict() },  path + '/checkpoint.tar')
