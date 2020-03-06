import numpy as np
import pandas as pd
import math
import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from collections import OrderedDict
from datetime import datetime
from PIL import Image


def Binarize(tensor):
    return tensor.sign()
    
def Ternarize(tensor, threshold):
    with torch.no_grad():
        return torch.where(torch.abs(tensor)> threshold, Binarize(tensor), torch.zeros_like(tensor))

class BinarizeLinear(torch.nn.Linear):

    def __init__(self, threshold, act_quant, ber, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

        self.threshold = threshold
        self.act_quant = act_quant
        self.ber = ber
        
    def forward(self, input):
        if self.act_quant=='B':
            input.data = Binarize(input.data)
        elif self.act_quant=='T':
            input.data = Ternarize(input.data, self.threshold)
        elif self.act_quant=='F':
             pass
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        #ERRORS : 1 to -1 and -1 to 1
        errors = torch.where(torch.rand_like(self.weight.data)>self.ber, torch.ones_like(self.weight.data), -torch.ones_like(self.weight.data))    
        self.weight.data=Binarize(self.weight.org)*errors

#         #NO ERRORS
#         self.weight.data=Binarize(self.weight.org)
        
        out = torch.nn.functional.linear(input, self.weight)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class TernarizeLinear(torch.nn.Linear):

    def __init__(self, threshold, act_quant, ber, *kargs, **kwargs):
        super(TernarizeLinear, self).__init__(*kargs, **kwargs)
        
        self.threshold = threshold
        self.act_quant = act_quant
        self.ber = ber
        
    def forward(self, input):

        if self.act_quant=='B':
            input.data = Binarize(input.data)
        elif self.act_quant=='T':
            input.data = Ternarize(input.data, self.threshold)
        elif self.act_quant=='F':
            pass

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        
        
#         #ERRORS : 1 to -1 and -1 to 1
#         errors = torch.where(torch.rand_like(self.weight.data)>self.ber, torch.ones_like(self.weight.data), -torch.ones_like(self.weight.data))    
#         self.weight.data=Ternarize(self.weight.org, self.threshold)*errors        

        #ERRORS : 0 to 1 or -1, 1 to 0 or -1 to 0
        self.weight.data=Ternarize(self.weight.org, self.threshold)
        p_0 = torch.where(self.weight.data==0.0, torch.ones_like(self.weight.data), torch.zeros_like(self.weight.data))
        p_0 = p_0.sum().item()/self.weight.data.numel()
        pm_ones = torch.where(torch.rand_like(self.weight.data)<0.5, torch.ones_like(self.weight.data), -torch.ones_like(self.weight.data))
        errors = torch.where(torch.rand_like(self.weight.data)>(2.0*self.ber)/(1+p_0), torch.zeros_like(self.weight.data), pm_ones)
        self.weight.data=Ternarize(self.weight.data+errors, self.threshold)          
        
#         #NO ERRORS
#         self.weight.data=Ternarize(self.weight.org, self.threshold)        
        
        out = torch.nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(torch.nn.Conv2d):

    def __init__(self, threshold, act_quant, ber, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

        self.threshold = threshold
        self.act_quant = act_quant
        self.ber = ber
        
    def forward(self, input):
        if input.size(1) != 3:
            if self.act_quant=='B':
                input.data = Binarize(input.data)
            elif self.act_quant=='T':
                input.data = Ternarize(input.data, self.threshold)
            elif self.act_quant=='F':
                pass
           
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        #ERRORS : 1 to -1 and -1 to 1
        errors = torch.where(torch.rand_like(self.weight.data)>self.ber, torch.ones_like(self.weight.data), -torch.ones_like(self.weight.data))    
        self.weight.data=Binarize(self.weight.org)*errors
        
#         #NO ERRORS
#         self.weight.data=Binarize(self.weight.org)
        
        out = torch.nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class TernarizeConv2d(torch.nn.Conv2d):

    def __init__(self, threshold, act_quant, ber, *kargs, **kwargs):
        super(TernarizeConv2d, self).__init__(*kargs, **kwargs)
        
        self.threshold = threshold
        self.act_quant = act_quant
        self.ber = ber
        
    def forward(self, input):
        if input.size(1) != 3:
            if self.act_quant=='B':
                input.data = Binarize(input.data)
            elif self.act_quant=='T':
                input.data = Ternarize(input.data, self.threshold)
            elif self.act_quant=='F':
                pass

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

#         #ERRORS : 1 to -1 and -1 to 1
#         errors = torch.where(torch.rand_like(self.weight.data)>self.ber, torch.ones_like(self.weight.data), -torch.ones_like(self.weight.data))    
#         self.weight.data=Ternarize(self.weight.org, self.threshold)*errors   
    
        #ERRORS : 0 to 1 or -1, 1 to 0 or -1 to 0
        self.weight.data=Ternarize(self.weight.org, self.threshold)
        p_0 = torch.where(self.weight.data==0.0, torch.ones_like(self.weight.data), torch.zeros_like(self.weight.data))
        p_0 = p_0.sum().item()/self.weight.data.numel()
        pm_ones = torch.where(torch.rand_like(self.weight.data)<0.5, torch.ones_like(self.weight.data), -torch.ones_like(self.weight.data))
        errors = torch.where(torch.rand_like(self.weight.data)>(2.0*self.ber)/(1+p_0), torch.zeros_like(self.weight.data), pm_ones)
        self.weight.data=Ternarize(self.weight.data+errors, self.threshold)             

#         #NO ERRORS
#         self.weight.data = Ternarize(self.weight.org, self.threshold)

        out = torch.nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

def normal_init(m):
    if m.__class__.__name__.find('Batch')!=-1:
        if m.weight is not None:
            torch.nn.init.ones_(m.weight)
    elif m.__class__.__name__.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, mean = 0, std = 0.03)
    elif m.__class__.__name__.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, mean = 0, std = 0.03)


def Conv(in_channels, out_channels, weight_quant, act_quant, threshold=None, ber=0.0):
    if weight_quant=='F':
        return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False)
    elif weight_quant=='B':
        return BinarizeConv2d(threshold, act_quant, ber, in_channels, out_channels, kernel_size=3, padding=1, bias = False)
    elif weight_quant=='T':
        return TernarizeConv2d(threshold, act_quant, ber, in_channels, out_channels, kernel_size=3, padding=1, bias = False)


def Linear(in_size, out_size, weight_quant, act_quant, threshold=None, ber=0.0):
    if weight_quant=='F':
        return torch.nn.Linear(in_size, out_size, bias=False)
    elif weight_quant=='B':
        return BinarizeLinear(threshold, act_quant, ber, in_size, out_size, bias=False)
    elif weight_quant=='T':
        return TernarizeLinear(threshold, act_quant, ber, in_size, out_size, bias=False)


class VGG(torch.nn.Module):
    def __init__(self, conv_list, fc_list, weight_quant, act_quant, ber=0.0, aff=False, threshold=None):
        super(VGG, self).__init__()

        self.ber = ber
        self.weight_quant = weight_quant
        self.act_quant = act_quant
        self.threshold = threshold
        self.features, out_size = self._make_conv(conv_list, aff)
        self.classifier = self._make_fc(out_size, fc_list, aff) 

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_conv(self, conv_list, aff):
        layers = []
        in_channels = 3
        data_size = 32
        for x in conv_list:
            if x == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
                data_size = int( 1+(data_size-2)/2 )
            elif x == 'B':
                layers += [torch.nn.BatchNorm2d(in_channels, affine=aff)]
                if self.weight_quant=='F' or self.act_quant=='F':
                    layers += [torch.nn.ReLU(inplace=True)]
                else:
                    layers += [torch.nn.Hardtanh(inplace=True)]
            elif x == 'D':
                layers += [torch.nn.Dropout(0.3)]
            elif isinstance(x, int):
                layers += [Conv(in_channels, x, self.weight_quant, self.act_quant, self.threshold, self.ber)]
                in_channels = x
        out_size = in_channels * data_size * data_size
        #layers += [torch.nn.AvgPool2d(kernel_size=1, stride=1)]
        return torch.nn.Sequential(*layers), out_size

    def _make_fc(self, out_size, fc_list, aff):
        layers = []
        in_size = out_size
        for x in fc_list:
            if isinstance(x, int):
                layers += [Linear(in_size, x, self.weight_quant, self.act_quant, self.threshold, self.ber), torch.nn.BatchNorm1d(x, affine=aff)]
                if self.weight_quant=='F' or self.act_quant=='F':
                    layers += [torch.nn.ReLU(inplace=True)]
                else:
                    layers += [torch.nn.Hardtanh(inplace=True)]
                in_size = x
            elif x == 'D':
                layers += [torch.nn.Dropout(0.5)]
        layers += [Linear(in_size, 10, self.weight_quant, self.act_quant, self.threshold, self.ber), torch.nn.BatchNorm1d(10, affine=aff)]
        return torch.nn.Sequential(*layers)


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss



def train(model, train_loader, optimizer, device, criterion = torch.nn.CrossEntropyLoss() ):
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # This loop is for Binary and Ternary parameters having 'org' attribute
        for p in list(model.parameters()): # blocking weights with org value greater than a threshold by setting grad to 0 
            if hasattr(p,'org'):
                p.data.copy_(p.org)
                
        optimizer.step()
        
        # This loop is only for Binary and Ternary parameters as they have 'org' attribute
        for p in list(model.parameters()):  # updating the org attribute
            if hasattr(p,'org'):
                p.org.copy_(p.data)


def test(model, loader, device, frac = 1, criterion = torch.nn.CrossEntropyLoss(), verbose = False):
    
    model.eval()
    loss = 0
    correct = 0
    
    for data, target in loader:
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)
        output = model(data)
        loss += criterion(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss = round( (loss * frac)/len(loader.dataset), 7)
    acc = round( 100. * float(correct) * frac / len(loader.dataset)  , 2)
    
    return acc, loss

def plot_parameters(model, path, save=True):
    
    fig = plt.figure(figsize=(15, 30))
    i = 1

    for (n, p) in model.named_parameters():
        
        if (n.find('bias') == -1) and (len(p.size()) != 1):  #bias or batchnorm weight -> no plot
            fig.add_subplot(8,2,i)
            if model.__class__.__name__.find('B') != -1:  #BVGG -> plot p.org
                if hasattr(p,'org'):
                    weights = p.org.data.cpu().numpy()
                else:
                    weights = p.data.cpu().numpy()
                binet = 100
            else:
                weights = p.data.cpu().numpy()            #TVGG or FVGG plot p
                binet = 50
            i+=1
            plt.title( n.replace('.','_') )
            plt.hist( weights.flatten(), binet)

    if save:
        time = datetime.now().strftime('%H-%M-%S')
        fig.savefig(path+'/'+time+'_weight_distribution.png')
    plt.close()


def plot_acc(path, data):
    df = pd.read_csv(path+'/'+data)
    x = df['epoch']
    for acc in ['acc_test']:
        y = df[acc]
        plt.figure(figsize=(14,12))
        plt.plot(x,y)
        plt.ylim((40,100))
        plt.yticks([40,60,80,90,100])
        plt.xlabel('Epoch')
        plt.ylabel(acc+'(\%)')
        plt.grid()
        plt.savefig(path + '/'+acc+'.png', format = 'png')
        plt.close()

def plot_loss(path, data):
    df = pd.read_csv(path+'/'+data)
    x = df['epoch']
    y_1 = df['tr_loss']
    y_2 = df['val_loss']
    plt.figure(figsize=(14,12))

    plt.plot(x, y_1, label='Train Loss')
    plt.plot(x, y_2, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid() 
    plt.savefig(path + '/loss.png', format = 'png')
    plt.close()   

def createHyperparametersFile(path, args):

    hyperparameters = open(path + r"/hyperparameters.txt","w+")
    L = ["- weight quantization: {}".format(args.weight_quant) + "\n",
        "- activation quantization: {}".format(args.act_quant) + "\n",
        "- lr: {}".format(args.lr) + "\n",
        "- optim: {}".format(args.optim) + "\n",
        "- threshold: {}".format(args.threshold) + "\n",
        "- MiniBatchSize: {}".format(args.mbs) + "\n",
        "- L2 Decay: {}".format(args.decay) + "\n",
        "- Bit Error Rate: {}".format(args.ber) + "\n",
        "- epochs: {}".format(args.epochs) + "\n"]
   
    hyperparameters.writelines(L)
    hyperparameters.close()



