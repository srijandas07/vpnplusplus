import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('--path', default='', type=str)

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
from dataset_test import *


def init_i3d():
    #INITIALIZE I3D WITH PRE-TRAINED WEIGHTS
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(31)
    #print(i3d)
    i3d.load_state_dict(torch.load('weights/weights_i3d.pt'))
    #summary(i3d, (3, 64, 224, 224))
    i3d.fuse_bin = True
    i3d.fuse_layer()
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    #summary(i3d, (3, 64, 224, 224))
    return i3d


def run(init_lr=0.01, max_steps=1, mode='rgb', root='/data/stars/user/rdai/smarthomes/Blurred_smarthome_clipped_SSD/', batch_size=16, path=''):
    # setup dataset
    #train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
    #                                       videotransforms.RandomHorizontalFlip(),
    #])
    i3d = init_i3d()
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset('./test_Labels.csv', 'test', root, 'rgb', test_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=36, pin_memory=True)

    #val_dataset = Dataset('/data/stars/user/sdas/smarthomes_data/splits/validation_new_CS.txt', 'val', root, 'rgb', test_transforms)
    #val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)    

    #dataloaders = {'train': dataloader, 'val': val_dataloader}
    #datasets = {'train': dataset, 'val': val_dataset}
    dataloaders = {'test': dataloader}
    datasets = {'test': dataset}
    
    # setup the model
    '''
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(31)
    '''
    #print(i3d)
    #summary(i3d, (3, 64, 224, 224))
    #i3d.replace_logits(157)
    checkpoint = torch.load('./weights_fused_new/{}'.format(path))
    #state_dict =checkpoint['state_dict']
    from collections import OrderedDict
    new_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
         name = 'module.'+key # remove module.
         new_checkpoint[name] = value
    #i3d.load_state_dict(torch.load('./weights_fused/{}'.format(path)))
    i3d.load_state_dict(new_checkpoint)
    '''
    model_pretrained = torch.load_state_dict('./weights_fused/{}'.format(path))
    pretrained_dict = model_pretrained.state_dict()
    model_dict = i3d.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    i3d.load_state_dict(torch.load(model_dict))
    '''

    i3d.cuda()
    #summary(i3d, (3, 64, 224, 224))
    #i3d = nn.DataParallel(i3d)
    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    #lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
    pred_arr = np.zeros((len(dataset), 31))

    num_steps_per_update = 1 # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print ('Step {}/{}'.format(steps, max_steps))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['test']:
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
            bal_dict = Bal_Dict()
            acount = 0
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                attention_weights, per_frame_logits, fused_flogits = i3d(inputs)
                #upsample to input size
                #print(per_frame_logits.shape, labels.shape)
                #per_frame_logits_up = F.upsample(per_frame_logits, t, mode='linear')
                #print(per_frame_logits.shape)
                #compute localization loss
                #loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                #tot_loc_loss += loc_loss.data
                
                #compute classification loss (with max-pooling along time B x C x T)
                #criterion=nn.CrossEntropyLoss().cuda()
                #cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], labels)
                #tot_cls_loss += cls_loss.data

                #loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                #loss = cls_loss
                #tot_loss += loss.data
                #loss.backward()
                #print(torch.max(labels, dim=1)[1])
                y_true = torch.max(per_frame_logits, dim=1)[1]
                #print(y_true.squeeze().tolist())
                for count in range(len(y_true.squeeze().tolist())):
                    l = per_frame_logits[count,:].cpu().detach().numpy()
                    pred_arr[acount,:] = l
                    acount+=1

                acc = calculate_accuracy(per_frame_logits, torch.max(labels, dim=1)[1])
                bal_dict.bal_update(per_frame_logits, torch.max(labels, dim=1)[1])
                #print(acc)
                tot_acc += acc
            steps += 1
            np.save("pred_arr.txt", pred_arr)
            if phase == 'test':
                print ('{} Acc: {:.4f}, Bal_acc: {:.4f}'.format(phase, tot_acc/num_iter, bal_dict.bal_score()))
    


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, path=args.path, batch_size=8)
