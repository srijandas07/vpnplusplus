import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
from _name_to_int import _name_to_int
import numpy as np
from random import randint
import json
import csv
import h5py
import random
import os
import os.path
import glob
import cv2
from feeders import tools
import pickle


class Bal_Dict():

    def __init__(self):
        self.balanced_dict = {}
        self.total_dict = {}

    def bal_update(self, outputs, y_true):
        #print(y_true, y_pred, type(y_true))
        l =len(y_true)
        y_pred = torch.max(outputs, dim=1)[1]
        y_pred = y_pred.squeeze().tolist()
        y_true = y_true.squeeze().tolist()
        #print(y_pred, y_true)
        for i in range(l):
            if y_true[i] in self.balanced_dict.keys():
                self.balanced_dict[y_true[i]]+=float(y_true[i]==y_pred[i])
            else:
                self.balanced_dict[y_true[i]]=(y_true[i]==y_pred[i])
            if y_true[i] in self.total_dict.keys():
                self.total_dict[y_true[i]]+=1.0
            else:
                self.total_dict[y_true[i]]=1.0

    def bal_score(self):
        count=0.0
        for i in self.total_dict.keys():
            if i in self.balanced_dict.keys():
                count+= ((1.0*self.balanced_dict[i])/self.total_dict[i])
        print(self.total_dict)
        print(len(self.total_dict.keys()))
        return count / len(self.total_dict.keys())

def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)
        #print(torch.max(outputs, dim=1)[0])
        #print(torch.max(outputs, dim=1)[1])
        #pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = torch.max(outputs, dim=1)[1]
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        #print(pred, targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()
        #print(n_correct_elems, batch_size)
        return n_correct_elems / batch_size

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


#def load_rgb_frames(image_dir, vid, start, num, nf):
def load_rgb_frames(frames):
  #frames = []
  images = []
  #print(image_dir, vid)
  #imgs = sorted(glob.glob(image_dir+vid+ "/*"))
  for i in frames:
    img=cv2.imread(i)[:, :, [2, 1, 0]]
    #img=cv2.imread(i)
    #img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    #img = (img/255.)*2 - 1
    images.append(img)
  images = np.asarray(images, dtype=np.float32)
  images = (images/127.5) - 1
  #while(len(frames)<num):
  #  frames.extend(images)
  #frames = frames[:num]
  return images

def load_flow_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
    
    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=31):
    dataset = []
    with open(split_file, 'r') as f:
        data = [os.path.splitext(i.strip())[0] for i in f.readlines()]

    i = 0
    for vid in data:
        #if data[vid]['subset'] != split:
        #    continue

        #if not os.path.exists(os.path.join(root, vid)):
        #    continue
        num_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == 'flow':
            num_frames = num_frames//2
            
        #if num_frames < 66:
        #    continue

        #label = np.zeros((num_classes,num_frames), np.float32)
        label=np.eye(num_classes)[_name_to_int(vid.split('_')[0], 'CS') - 1]
        #fps = num_frames/data[vid]['duration']
        #for ann in data[vid]['actions']:
        #    for fr in range(0,num_frames,1):
        #        if fr/fps > ann[1] and fr/fps < ann[2]:
        #            label[ann[0], fr] = 1 # binary classification
        dataset.append((vid, label, num_frames))
        i += 1
    
    return dataset


class Dataset(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, data_path='./', label_path='./',
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.sample_duration = 64
        self.step = 2

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, nf = self.data[index]
        #start_f = random.randint(0,nf-64)
        frame_indices = []
        images = sorted(glob.glob(self.root + vid + "/*"))
        n_frames=len(images)

        if (n_frames > self.sample_duration * self.step):
            start = randint(0, n_frames - self.sample_duration*self.step)
            for i in range(start, start + self.sample_duration*self.step, self.step):
                frame_indices.append(images[i])
        elif n_frames < self.sample_duration:
            while len(frame_indices) < self.sample_duration:
                frame_indices.extend(images)
            frame_indices = frame_indices[:self.sample_duration]
        else:
            start = randint(0, n_frames - self.sample_duration)
            for i in range(start, start+self.sample_duration):
                frame_indices.append(images[i])

        if self.mode == 'rgb':
            imgs = load_rgb_frames(frame_indices)
        else:
            imgs = load_flow_frames(self.root, vid, start_f, 64)
        #label = label[:, start_f:start_f+64]

        #imgs = self.transforms(imgs)
        #now for skeleton

        data_numpy = self.data_skl[index]
        label_skl = self.label_skl[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return video_to_tensor(imgs), torch.from_numpy(label), data_numpy, label_skl


    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label_skl = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label_skl = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data_skl = np.load(self.data_path, mmap_mode='r')
        else:
            self.data_skl = np.load(self.data_path)
        if self.debug:
            self.label_skl = self.label[0:100]
            self.data_skl = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data_skl = self.data_skl
        N, C, T, V, M = data_skl.shape
        self.mean_map = data_skl.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data_skl.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.data)
