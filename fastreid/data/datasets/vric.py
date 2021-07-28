# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import glob
import os.path as osp
import re
import pdb
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VRIC(ImageDataset):
    dataset_dir = 'VRIC'
    dataset_url = None
    dataset_name = "VRIC"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        
        train = self._process_train_dir('vric_train.txt', 'train_images' )
        query = self._process_test_dir('vric_probe.txt', 'probe_images' )
        gallery = self._process_test_dir('vric_gallery.txt', 'gallery_images' )

        super(VRIC, self).__init__(train, query, gallery, **kwargs)

    def _process_train_dir(self, txt_file, img_fold):
        dataset = []
        with open(osp.join(self.dataset_dir, txt_file),'r') as f:
           lines = f.readlines()
           for line in lines:
              line = line[:-1]
              img_path, pid, camid= line.split(' ')
              img_path = osp.join(self.dataset_dir, img_fold+'/'+ img_path)
              if int(pid) == -1: continue  # junk images are just ignored
              pid = self.dataset_name + "_" + pid
              camid = self.dataset_name + "_" + camid
              dataset.append((img_path, pid, camid))
        
        return dataset
        
    def _process_test_dir(self, txt_file, img_fold):
        dataset = []
        with open(osp.join(self.dataset_dir, txt_file),'r') as f:
           lines = f.readlines()
           for line in lines:
              line = line[:-1]
              img_path, pid, camid= line.split(' ')
              img_path = osp.join(self.dataset_dir, img_fold+'/'+img_path) 
              dataset.append((img_path, int(pid), int(camid)))
        return dataset      