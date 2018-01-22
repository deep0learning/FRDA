'''
    implement domain adversarial training process 
    @author: liuxin
    @date: 2018.1.03
'''
from __future__ import print_function
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

from DA_network  import Network_DA

from load_imglist import ImageList

parser = argparse.ArgumentParser(description='Domain adversarial Training')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr_1', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='domain critic learning rate')
parser.add_argument('--lr_2', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='cls and feature learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')

parser.add_argument('--FExtractor', default='LightCNN-9', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29')

parser.add_argument('--critic', default='wasserstein', type=str, metavar='Model',
                    help='model type: cross-entropy or wasserstein')
parser.add_argument('--critic_steps', default='', type=int, metavar='training steps',
                    help='training steps :steps for training critic network ')
parser.add_argument('--gamma', default='', type=float, metavar='coefficient',
                    help='balancing coefficients :balance wasserstein distance and grad loss ')

parser.add_argument('--classifier', default='softmax', type=str, metavar='Model',
                    help='model type: softmax,A_softmax')
parser.add_argument('--num_classes', default='', type=int, metavar='classes',
                    help='source classifier :class numbers')
parser.add_argument('--lambda', default='', type=float, metavar='coefficient',
                    help='balancing coefficients :balance cls loss  and grad wasserstein loss ')


parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--epoch', default='', type=int, metavar='epoch to resume',
                    help='epoch number to resume training ')

parser.add_argument('--root_path', default='', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')

parser.add_argument('--train_list', default='', type=str, metavar='PATH',
                    help='path to training list (default: none)')

parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')

def main():
    global args
    args = parser.parse_args()

    '''
    basemodel:LightCNN9 or LightCNN29  or other network

    discriminator:discriminate src or target features
    metric loss ；cross entropy or wassertein metric on src and target feature discrepancy

    source classifier:softmax or A_softmax
    
    '''
    model = Network_DA(args.critic,args.classifier,args.FExtractor)
    if args.cuda:
        #parallel model
        model = torch.nn.DataParallel(model,device_ids=[2,3]).cuda()
    print(model)


    '''
    construct an optimizer,do not include critic network optimize
    '''            
    optimizer = torch.optim.Adam(model.FExtractor.parameters(), args.lr_2,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cls_optim = torch.optim.Adam(model.classifier.parameters(),args.lr_2)
    critic_optim = torch.optim.Adam(model.critic.parameters(),args.lr_1)
    '''
    resume training	
    '''		
    if args.resume:
       load_network(model,'all',args.epoch,args.save_dir)


    '''
    @brief:load images from different domains and corresponding labels if any
    train_loader:batch( nir and vis images)
    '''
    train_loader = torch.utils.data.DataLoader(
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    '''
    begin tarining	
    '''	
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer,critic_optim,cls_optim, epoch)

        # train for one epoch
        train(train_loader, model, optimizer,critic_optim,cls_optim,epoch)

        save_name = args.save_path + 'DA_' + str(epoch+1) + '_checkpoint.pth.tar'
        
        save_checkpoint({}, save_name)


def train(train_loader, model, criterion, optimizer,critic_optim,cls_optim, epoch):
    
    end = time.time()
    # each time read a batch containing source sample and target samples
    for i, (input, clslabel,domain_label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input      = input.cuda()
        clslabel     = clslabel.cuda()
        domain_label = domain_label.cuda()

        input_var  = torch.autograd.Variable(input)
        clslabel_var = torch.autograd.Variable(clslabel)
        domain_label_var = torch.autograd.Variable(domain_label)
        
	#feature extraction
	f_out = model.FExtractor(input_var)


	'''
	separate source and target features from a feature batch
	'''
	src,src_label,target,target_label = Data_separate(f_out,clslabel_var,domain_label_var)


        '''
        train domain critic optimally with wassertein losses or entropy loss
        '''
        #train domain critic to optimally and compute losses when optimally
	critic_dict={'steps':args.step,'gamma':args.gamma}
        critic_loss = critic_model_train(model.critic,src,target,critic_optim,critic_dict)
	
	#classifier loss from source sample features
	src_pred = model.classifier(src,args.num_classes)
        cls_loss =  nn.CrossEntropyLoss(src_pred,src_label)
	cls_optim.zero_grad()
        cls_loss.backward()
	cls_optim.step()

	cls_loss,wd_loss = model(src,srclabel,target)
	total_loss = cls_loss + args.lambda*wd_loss
        optimizer.zero_grad()
	total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time ({batch_time:.3f})\t'
                  'Data ({data_time:.3f})\t'
                  'Loss ({loss:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

## save model
def save_checkpoint(state, filename):
    torch.save(state, filename)


'''
adjust lerning rate for faeture extractor
'''
def adjust_learning_rate(optimizer, epoch):
	pass
	
if __name__ == '__main__':
    main()
