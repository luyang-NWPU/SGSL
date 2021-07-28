# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import glob
import os.path as osp
import re
import pdb
from .bases import ImageDataset, get_pid_camid, load_pickle
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Vehicle1M(ImageDataset):
    dataset_dir = 'Vehicle-1M'
    dataset_url = None
    dataset_name = "Vehicle1M"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        
        train_im_names, test_im_names = [], []
        train_im_ids, test_im_ids = [], []
        
        train_path = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
        with open(train_path, 'r') as f:
           lines = f.readlines()
           for line in lines:
              name = line.split(' ')[0]
              id = int(line.split(' ')[1])
              im_path = osp.join(self.dataset_dir, 'image/' + name)
              train_im_names.append(im_path)
              train_im_ids.append(id)
        
        
        test_path = osp.join(self.dataset_dir, 'train_test_split/test_3000.txt')
        with open(test_path, 'r') as f:
           lines = f.readlines()
           for line in lines:
              name = line.split(' ')[0]
              id = int(line.split(' ')[1])
              im_path = osp.join(self.dataset_dir, 'image/' + name)
              test_im_names.append(im_path)
              test_im_ids.append(id)
                
        train = self._process_train_dir(train_im_names, train_im_ids)
        query, gallery = self._process_test_dir(test_im_names, test_im_ids)
        
        super(Vehicle1M, self).__init__(train, query, gallery, **kwargs)

    def _process_train_dir(self, im_names, ids):
        
        dataset = []
        for name, pid in zip(im_names, ids):
            camid = 0
            if pid == -1: continue  # junk images are just ignored
            pid = self.dataset_name + "_" + str(pid)
            camid = self.dataset_name + "_" + str(camid)
            dataset.append((name, pid, camid))
             
        return dataset
        
    def _process_test_dir(self, im_names, ids):
        
        query = []
        gallery = []
        query_path = osp.join(self.dataset_dir, 'train_test_split/Vehicle1M_query.txt')
        with open(query_path,'r') as f:
             lines = f.readlines()
             for line in lines:
                 items = line.split(' ')
                 query.append((items[0], int(items[1]), int(items[2])))
                 
            
        gallery_path = osp.join(self.dataset_dir, 'train_test_split/Vehicle1M_gallery.txt')
        with open(gallery_path,'r') as f:
             lines = f.readlines()
             for line in lines:
                 items = line.split(' ')
                 gallery.append((items[0], int(items[1]), int(items[2][:-1])))
        
        return query, gallery        
        
'''
    def _process_train_dir(self, im_names):
        
        dataset = []
        for name in im_names:
            pid, camid = get_pid_camid(name)   
            if pid == -1: continue  # junk images are just ignored
            pid = self.dataset_name + "_" + str(pid)
            camid = self.dataset_name + "_" + str(camid)
            img_path = osp.join(self.dataset_dir, 'images/'+name)
            dataset.append((img_path, pid, camid))
        return dataset
        
    def _process_test_dir(self, im_names, test_marks):
        query = []
        gallery = []
        if test_marks==None:
          gallery_ids = []
          random.shuffle(im_names)
          for name in im_names:
             img_path = osp.join(self.dataset_dir, 'images/'+name)
             pid, camid = get_pid_camid(name)
             if pid not in gallery_ids:
                gallery_ids.append(pid)
                gallery.append((img_path, pid, camid))
             else:
                query.append((img_path, pid, camid))
        else:
          for name, mark in zip(im_names, test_marks):
            pid, camid = get_pid_camid(name)
            img_path = osp.join(self.dataset_dir, 'images/'+name)
            if mark==0:
               query.append((img_path, pid, camid))
            elif mark==1:
               gallery.append((img_path, pid, camid))
            else:
               print('ERROR in VehicleID test set.')
               assert 0==1
          
        return query, gallery      
'''