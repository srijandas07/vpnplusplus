import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('--save_model', default='weights/', type=str)
parser.add_argument('--root', default='', type=str)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d
from torchsummary import summary
from dataset import *


def run(init_lr=0.1, max_steps=100, mode='rgb', root='/data/stars/user/rdai/smarthomes/Blurred_smarthome_clipped_SSD/', batch_size=1, save_model='weights_i3d/'):
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset('/data/stars/user/sdas/smarthomes_data/splits/train_new_CS.txt', 'train', root, 'rgb', train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)

    val_dataset = Dataset('/data/stars/user/sdas/smarthomes_data/splits/validation_new_CS.txt', 'val', root, 'rgb', test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    
    f = csv.writer(open('out.csv', 'w'), delimiter='\t')
    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(31)
    
    #print(i3d)
    #summary(i3d, (3, 64, 224, 224))
    #i3d.replace_logits(157)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    #summary(i3d, (3, 64, 224, 224))
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9)
    #lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [20, 50])
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

    num_steps_per_update = 1 # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print ('Step {}/{}'.format(steps, max_steps))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
            #print(phase)    
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            tot_acc = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                #t = inputs.size(2)
                labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs)
                #upsample to input size
                #print(per_frame_logits.shape, labels.shape)
                #per_frame_logits_up = F.upsample(per_frame_logits, t, mode='linear')
                #print(per_frame_logits.shape)
                #compute localization loss
                #loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                #tot_loc_loss += loc_loss.data
                
                #compute classification loss (with max-pooling along time B x C x T)
                criterion=nn.CrossEntropyLoss().cuda()
                #cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], labels)
                cls_loss = criterion(per_frame_logits, torch.max(labels, dim=1)[1].long())
                tot_cls_loss += cls_loss.data

                #loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                loss = cls_loss
                tot_loss += loss.data
                loss.backward()
                #print(torch.max(labels, dim=1)[1])
                #acc = calculate_accuracy(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=1)[1])
                acc = calculate_accuracy(per_frame_logits, torch.max(labels, dim=1)[1])
                #print(acc)
                tot_acc += acc
                if phase == 'train':
                    #steps += 1
                    #num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    #lr_sched.step()

            if phase == 'train':
                print ('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}, Acc: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, tot_loss/num_iter, tot_acc/num_iter))
                # save model
                torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                tot_loss = tot_loc_loss = tot_cls_loss = tot_acc = 0.
                steps += 1
            if phase == 'val':
                lr_sched.step(tot_cls_loss/num_iter)
                print ('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}, Acc: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, tot_acc/num_iter))
    


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, batch_size=16)
