# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os, pdb
import glob
import numpy as np
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset
from .bases import read_json, write_json

__all__ = ['VIPeR', ]


@DATASET_REGISTRY.register()
class VIPeR(ImageDataset):
    dataset_dir = "VIPeR"
    dataset_name = "viper"

    def __init__(self, root='datasets', split_id=0, **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.cam_a_dir = os.path.join(self.dataset_dir, 'cam_a')
        self.cam_b_dir = os.path.join(self.dataset_dir, 'cam_b')
        self.split_path = os.path.join(self.dataset_dir, 'splits.json')

        required_files = [self.dataset_dir, self.cam_a_dir, self.cam_b_dir]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, '
                'but expected between 0 and {}'.format(
                    split_id,
                    len(splits) - 1
                )
            )
        split = splits[split_id]

        train = split['train']
        query = split['query'] # query and gallery share the same images
        gallery = split['gallery']

        train = [(item[0],self.dataset_name+'_'+str(item[1]),self.dataset_name+'_'+str(item[2]) ) for item in train]
        super(VIPeR, self).__init__(train, query, gallery, **kwargs)

    def prepare_split(self):
        if not os.path.exists(self.split_path):
            print('Creating 10 random splits of train ids and test ids')

            cam_a_imgs = sorted(glob.glob(os.path.join(self.cam_a_dir, '*.bmp')))
            cam_b_imgs = sorted(glob.glob(os.path.join(self.cam_b_dir, '*.bmp')))
            assert len(cam_a_imgs) == len(cam_b_imgs)
            num_pids = len(cam_a_imgs)
            print('Number of identities: {}'.format(num_pids))
            num_train_pids = num_pids // 2
            """
            In total, there will be 20 splits because each random split creates two
            sub-splits, one using cameraA as query and cameraB as gallery
            while the other using cameraB as query and cameraA as gallery.
            Therefore, results should be averaged over 20 splits (split_id=0~19).
            
            In practice, a model trained on split_id=0 can be applied to split_id=0&1
            as split_id=0&1 share the same training data (so on and so forth).
            """
            splits = []
            for _ in range(10):
                order = np.arange(num_pids)
                np.random.shuffle(order)
                train_idxs = order[:num_train_pids]
                test_idxs = order[num_train_pids:]
                assert not bool(set(train_idxs) & set(test_idxs)), \
                    'Error: train and test overlap'

                train = []
                for pid, idx in enumerate(train_idxs):
                    cam_a_img = cam_a_imgs[idx]
                    cam_b_img = cam_b_imgs[idx]
                    train.append((cam_a_img, pid, 0))
                    train.append((cam_b_img, pid, 1))

                test_a = []
                test_b = []
                for pid, idx in enumerate(test_idxs):
                    cam_a_img = cam_a_imgs[idx]
                    cam_b_img = cam_b_imgs[idx]
                    test_a.append((cam_a_img, pid, 0))
                    test_b.append((cam_b_img, pid, 1))

                # use cameraA as query and cameraB as gallery
                split = {
                    'train': train,
                    'query': test_a,
                    'gallery': test_b,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

                # use cameraB as query and cameraA as gallery
                split = {
                    'train': train,
                    'query': test_b,
                    'gallery': test_a,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))
