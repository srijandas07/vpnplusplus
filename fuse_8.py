import torch
import torch.nn as nn
from pytorch_i3d import InceptionI3d
from pytorch_i3d_attn import InceptionI3d_skl
from torchsummary import summary
from dataset import *
from main import *
import numpy as np


class Cont_Loss(torch.nn.Module):

    def __init__(self):
        super(Cont_Loss, self).__init__()

    def forward(self, x, y, n_pos, n_neg):
        #y = y[:x.shape[0]]
        bs = x.shape[0]
        dist = torch.empty(bs)
        for i in range(bs):
            dist[i] = torch.exp(torch.matmul(
                x[i].reshape(1, -1), y[i].reshape(1, -1).t()))
        pos_target = torch.ones([n_pos])
        neg_target = torch.zeros([n_neg])
        target = torch.cat([pos_target, neg_target])
        return torch.nn.functional.binary_cross_entropy_with_logits(dist, target).cuda()


class Fusion_Loss(torch.nn.Module):

    def __init__(self):
        super(Fusion_Loss, self).__init__()

    def forward(self, x, y):
        y = y[:x.shape[0]]
        return torch.sum(torch.pow(torch.abs(torch.sub(x, y)), 2))


def init_i3d():
    # INITIALIZE I3D WITH PRE-TRAINED WEIGHTS
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(31)
    # print(i3d)
    i3d.load_state_dict(torch.load('weights/weights_i3d.pt'))
    #summary(i3d, (3, 64, 224, 224))
    i3d.fuse_bin = True
    i3d.fuse_layer()
    i3d.cuda()
    #summary(i3d, (3, 64, 224, 224))
    i3d = nn.DataParallel(i3d)
    #summary(i3d, (3, 64, 224, 224))
    return i3d


def init_i3d_skel():
    # INITIALIZE I3D WITH PRE-TRAINED WEIGHTS
    i3d = InceptionI3d_skl(400, in_channels=3)
    i3d.replace_logits(31)
    # print(i3d)
    i3d.load_state_dict(torch.load('weights/weights_i3d.pt'))
    i3d.fuse_layer()
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    #summary(i3d, (3, 64, 224, 224))
    return i3d


def init_agcn(flag):
    # INITIALIZE AGCN WITH PRE-TRAINED WEIGHTS
    if flag == 1:
        parser = get_parser1()
        print('attention')
    else:
        parser = get_parser2()
        print('contrastive')
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(12)
    proc = Processor(arg)
    agcn = proc.load_model()
    #summary(agcn, (3, 400, 15, 1))
    return proc, agcn


if __name__ == '__main__':
    i3d = init_i3d()
    i3d_skel = init_i3d_skel()
    proc, agcn = init_agcn(1)
    proc_cont, agcn_cont = init_agcn(2)
    batch_size = 16
    n_pos = 8
    n_neg = batch_size - n_pos
    weight_factor = 0.01
    split = 'new'
    root = '/data/stars/user/rdai/smarthomes/Blurred_smarthome_clipped_SSD/'
    dataset = Dataset('/data/stars/user/sdas/smarthomes_data/splits/train_'+split+'_CS.txt', 'train', root, 'rgb', None,
                      '/data/stars/user/sdas/PhD_work/poses_attention/2s-AGCN-For-Daily-Living/data/xsub/train_data_joint_'+split+'.npy',
                      '/data/stars/user/sdas/PhD_work/poses_attention/2s-AGCN-For-Daily-Living/data/xsub/train_label_'+split+'.pkl',
                      random_choose=4000, random_shift=False, random_move=False,
                      window_size=400, normalization=False, debug=False, use_mmap=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=n_pos, shuffle=True, num_workers=36, drop_last=True, pin_memory=True)

    val_dataset = Dataset('/data/stars/user/sdas/smarthomes_data/splits/validation_'+split+'_CS.txt', 'val', root, 'rgb', None,
                          '/data/stars/user/sdas/PhD_work/poses_attention/2s-AGCN-For-Daily-Living/data/xsub/val_data_joint_'+split+'.npy',
                          '/data/stars/user/sdas/PhD_work/poses_attention/2s-AGCN-For-Daily-Living/data/xsub/val_label_'+split+'.pkl',
                          random_choose=4000, random_shift=False, random_move=False,
                          window_size=400, normalization=False, debug=False, use_mmap=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=n_pos, shuffle=False, num_workers=36, drop_last=True, pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    ske_dataloader = proc_cont.load_data(n_neg)
    save_model = 'weights_fused_new/'
    max_steps = 100
    lr = 0.01
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9)
    #lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [20, 50])
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True)

    num_steps_per_update = 1  # accum gradient
    steps = 0

    # train it
    while steps < max_steps:  # for epoch in range(num_epochs):
        print ('Step {}/{}'.format(steps, max_steps))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
                i3d_skel.train(True)
                agcn.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                i3d_skel.train(False)
                agcn.train(False)
            # print(phase)
            tot_loss = 0.0
            tot_acc = 0.0
            num_iter = 0
            optimizer.zero_grad()
            #process = tqdm(ske_dataloader[phase])
            # Iterate over data.
            for neg_ske_data, data in tqdm(zip(ske_dataloader[phase], dataloaders[phase])):
                num_iter += 1
                # get the inputs
                #inputs, labels = i3d_data
                inputs, labels, ske_input, ske_label = data

                inputs = torch.cat([inputs, inputs])
                labels = torch.cat([labels, labels])
                ske_label = torch.cat([ske_label, ske_label])
                ske_input_att = torch.cat([ske_input, ske_input])
                ske_input_cont = torch.cat([ske_input, neg_ske_data])
                #print(torch.max(labels, dim=1)[1].long(), ske_label)
                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                attention_weights, per_frame_logits, fuse_logits = i3d(inputs)
                ske_input_att = Variable(ske_input_att.float().cuda())
                ske_label = Variable(ske_label.long().cuda())
                ske_output, ske_emb = agcn(ske_input_att)
                per_frame_logits_skel = i3d_skel(inputs, ske_output)
                #print(per_frame_logits.shape, fuse_logits.shape)
                # Skeleton model output
                #ske_input, ske_label, index = ske_data
                with torch.no_grad():
                    ske_input_cont = Variable(
                        ske_input_cont.float().cuda(),
                        requires_grad=False)
                    ske_label = Variable(
                        ske_label.long().cuda(),
                        requires_grad=False)
                    ske_output = agcn_cont(ske_input_cont)
                #print(fuse_logits.shape, ske_output.shape)
                # compute classification loss (with max-pooling along time B x C x T)
                criterion = nn.CrossEntropyLoss().cuda()

                loss1 = criterion(per_frame_logits, torch.max(
                    labels, dim=1)[1].long())
                loss2 = criterion(per_frame_logits_skel,
                                  torch.max(labels, dim=1)[1].long())
                tot_loss += loss1.data
                tot_loss += 0.1*loss2.data

                # INCLUDE FUSION LOSS
                fusion_loss = Fusion_Loss().cuda()
                fuse_loss = fusion_loss(attention_weights, ske_emb)

                cont_loss = Cont_Loss().cuda()
                cont_loss = cont_loss(fuse_logits, ske_output, n_pos, n_neg)

                #fuse_loss = torch.sum(torch.square(torch.abs(torch.sub(per_frame_logits, ske_output))))
                # print(fuse_loss)
                tot_loss += fuse_loss.data * 0.001
                t_loss = (0.9*loss1) + (0.1*loss2) + \
                    (fuse_loss*0.001) + (0.001*cont_loss)
                t_loss.backward()

                acc = calculate_accuracy(
                    per_frame_logits, torch.max(labels, dim=1)[1])
                '''
                if phase == 'val':
                    print(torch.max(per_frame_logits, dim=1)[1].long(), torch.max(labels, dim=1)[1].long())
                '''
                # print(acc)
                tot_acc += acc
                if phase == 'train':
                    optimizer.step()
                    optimizer.zero_grad()
                    # lr_sched.step()

            if phase == 'train':
                print ('{} Tot Loss: {:.4f}, Acc: {:.4f}'.format(
                    phase, tot_loss/num_iter, tot_acc/num_iter))
                # save model
                torch.save(i3d.module.state_dict(),
                           save_model+str(steps).zfill(6)+'.pt')
                tot_loss = tot_acc = 0.
                steps += 1
            if phase == 'val':
                lr_sched.step(tot_loss/num_iter)
                print ('{} Tot Loss: {:.4f}, Acc: {:.4f}'.format(
                    phase, (tot_loss)/num_iter, tot_acc/num_iter))
