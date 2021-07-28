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
class VRAI(ImageDataset):
    dataset_dir = 'VRAI'
    dataset_url = None
    dataset_name = "VRAI"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        
        pkl_path = osp.join(self.dataset_dir, 'partitions.pkl')
        partitions = load_pickle(pkl_path)
        train_im_names = partitions['train_im_names']
        test_im_names = partitions['test_im_names']
        test_marks = partitions['test_marks']
        
        
        train = self._process_train_dir(train_im_names)
        query, gallery = self._process_test_dir(test_im_names, test_marks=test_marks)
        
        super(VRAI, self).__init__(train, query, gallery, **kwargs)

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
'''