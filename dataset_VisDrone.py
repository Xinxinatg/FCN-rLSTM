import os
import xml.etree.ElementTree as ET

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from skimage import io
import skimage.transform as SkT
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import np_transforms as NP_T
from utils import density_map

class VisDrone(Dataset):
    r"""
    Wrapper for the TRANCOS dataset, presented in:
    Guerrero-Gómez-Olmedo et al., "Extremely overlapping vehicle counting.", IbPRIA 2015.
    """

    def __init__(self, train=True, path='./TRANCOS_v3', out_shape=(120, 160), transform=None, gamma=2.5, get_cameras=False, cameras=None, load_all=True):
        r"""
        Args:
            train: train (`True`) or test (`False`) images (default: `True`).
            path: path for the dataset (default: "./TRANCOS_v3").
            out_shape: shape of the output images (default: (120, 176)).
            transform: transformations to apply to the images as np.arrays (default: None).
            gamma: precision parameter of the Gaussian kernel (default: 30).
            get_cameras: whether or not to return the camera ID of each image (default: `False`).
            cameras: list with the camera IDs to be used, so that images from other cameras are discarded;
                if `None`, all cameras are used; it has no effect if `get_cameras` is `False` (default: `None`).
        """
        self.path = path
        self.out_shape = out_shape
        self.transform = transform
        self.gamma = gamma
        self.load_all = load_all

        if train:  # train + validation
            self.image_files = [img[:-1] for img in open(os.path.join(self.path, 'image_set', 'trainval.txt'))]
        else:  # test
            self.image_files = [img[:-1] for img in open(os.path.join(self.path, 'image_set', 'test.txt'))]

        self.cam_ids = {}
        if get_cameras:
            with open(os.path.join(self.path, 'cam_annotations.txt')) as f:
                for line in f:
                    img_f, cid = line.split()
                    if img_f in self.image_files:
                        self.cam_ids[img_f] = int(cid)

            if cameras is not None:
                # only keep images from the provided cameras
                self.image_files = [img_f for img_f in self.image_files if self.cam_ids[img_f] in cameras]
                self.cam_ids = {img_f: self.cam_ids[img_f] for img_f in self.image_files}

        # get the coordinates of the centers of all vehicles in all images
        self.centers = {img_f: [] for img_f in self.image_files}
        for img_f in self.image_files:
            mat = scipy.io.loadmat(os.path.join(self.path, 'images',img_f.replace('.jpg','.mat').replace('IMG_','GT_IMG_')))
            gt = mat["image_info"][0,0][0,0][0]
            x = np.array([[1,1]])
            number=mat["image_info"][0,0][0,0][0].shape[0]
            b=np.repeat(x, [number], axis=0)       
            new_gt=gt-b
            self.centers[img_f]=new_gt

        if self.load_all:
            # load all the data into memory
            self.images, self.densities = [], []
            for img_f in self.image_files:
                X, density = self.load_example(img_f)
                self.images.append(X)                
                self.densities.append(density)

    def load_example(self, img_f):
        # load the image 
        X = io.imread(os.path.join(self.path, 'images', img_f))
        img_centers = self.centers[img_f]

        # reduce the size of image by the given amount
        H_orig, W_orig = X.shape[0], X.shape[1]
        if H_orig != self.out_shape[0] or W_orig != self.out_shape[1]:
            X = SkT.resize(X, self.out_shape, preserve_range=True).astype('uint8')

        # compute the density map
        density = density_map(
            (H_orig, W_orig),
            img_centers,
            self.gamma*np.ones((len(img_centers), 2)),
            out_shape=self.out_shape)
        density = density[:, :, np.newaxis].astype('float32')

        return X, density

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        r"""
        Returns:
            X: image.
            density: vehicle density map.
            count: number of vehicles in the masked image.
            cam_id: camera ID (only if `get_cameras` is `True`).
        """
        if self.load_all:
            img_f = self.image_files[i]
            X = self.images[i]
            density = self.densities[i]
            img_centers = self.centers[img_f]
        else:
            img_f = self.image_files[i]
            X, density = self.load_example(img_f)
            img_centers = self.centers[img_f]

        # get the number of vehicles in the image and the camera ID
        count = len(img_centers)

        if self.transform:
            # apply the transformation to the image and density map
            X, density = self.transform([X, density])

        if self.cam_ids:
            cam_id = self.cam_ids[img_f]
            return X, density, count, cam_id
        else:
            return X, density, count

class VisDroneSeq(VisDrone):
    r"""
    Wrapper for the TRANCOS dataset, presented in:
    Guerrero-Gómez-Olmedo et al., "Extremely overlapping vehicle counting.", IbPRIA 2015.
    This version assumes the data is sequential, i.e. it returns sequences of images captured by the same camera.
    """

    def __init__(self, train=True, path='./TRANCOS_v3', size_red=8, transform=NP_T.ToTensor(), gamma=30, max_len=None, cameras=None):
        r"""
        Args:
            train: train (`True`) or test (`False`) images (default: `True`).
            path: path for the dataset (default: "./TRANCOS_v3").
            out_shape: shape of the output images (default: (120, 176)).
            transform: transformations to apply to the images as np.arrays (default: `NP_T.ToTensor()`).
            gamma: precision parameter of the Gaussian kernel (default: 30).
            max_len: maximum sequence length (default: `None`).
            cameras: list with the camera IDs to be used, so that images from other cameras are discarded;
                if `None`, all cameras are used; it has no effect if `get_cameras` is `False` (default: `None`).
        """
        super(VisDroneSeq, self).__init__(train=train, path=path, out_shape=(120, 160), transform=transform, gamma=gamma, get_cameras=True, cameras=cameras)
        self.img2idx = {img: idx for idx, img in enumerate(self.image_files)}  # hash table from file names to indices
        self.seqs = []  # list of lists containing the names of the images in each sequence
        prev_cid = -1
        cur_len = 0
        with open('/content/VisDrone2020-CC-FCN-rLSTM/cam_annotations.txt') as f:
            for line in f:
                img_f, cid = line.split()
                if img_f in self.image_files:
                    # all images in the sequence must be from the same camera
                    # and all sequences must have length not greater than max_len
                    if (int(cid) == prev_cid) and ((max_len is None) or (cur_len < max_len)):
                        self.seqs[-1].append(img_f)
                        cur_len += 1
                    else:
                        self.seqs.append([img_f])
                        cur_len = 1
                        prev_cid = int(cid)

        if max_len is None:
            # maximum sequence length in the dataset
            self.max_len = max([len(seq) for seq in self.seqs])
        else:
            self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        r"""
        Returns:
            X: sequence of images, tensor with shape (max_seq_len, channels, height, width)
            density: sequence of vehicle density maps for each image, tensor with shape (max_seq_len, 1, height, width)
            count: sequence of vehicle counts for each image, tensor with shape (max_seq_len)
            cam_id: camera ID, integer
            seq_len: length of the sequence (before padding), integer
        """
        seq = self.seqs[i]
        seq_len = len(seq)

        # randomize the (random) transformations applied to the first image of the sequence
        # and then apply the same transformations to the remaining images of the sequence
        if isinstance(self.transform, T.Compose):
            for transf in self.transform.transforms:
                if hasattr(transf, 'rand_state'):
                    transf.reset_rand_state()
        elif hasattr(self.transform, 'rand_state'):
            self.transform.reset_rand_state()

        # build the sequences
        X = torch.zeros(self.max_len, 3, self.out_shape[0], self.out_shape[1])
        density = torch.zeros(self.max_len, 1, self.out_shape[0], self.out_shape[1])
        count = torch.zeros(self.max_len)
        for j, img_f in enumerate(seq):
            idx = self.img2idx[img_f]
            X[j], density[j], count[j], cam_id = super().__getitem__(idx)

        return X, density, count, cam_id, seq_len


train_transf = T.Compose([
        NP_T.RandomHorizontalFlip(0.5, keep_state=True),  # data augmentation: horizontal flipping (we could add more transformations)
        NP_T.ToTensor() ])
train_data = VisDroneSeq(train=True, path='/content/VisDrone2020-CC-FCN-rLSTM', size_red=8, transform=train_transf, gamma=10, max_len=5)
