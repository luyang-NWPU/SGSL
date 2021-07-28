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
class VD1(ImageDataset):
    dataset_dir = 'VD1'
    dataset_url = None
    dataset_name = "VD1"
    test_count = 0
    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        
        train = self._process_train_dir('train_test/trainlist.txt', 'image' )
        query = self._process_test_dir('query_ref/querylist.txt', 'image' )
        gallery = self._process_test_dir('query_ref/large_set.txt', 'image' )
        
        super(VD1, self).__init__(train, query, gallery, **kwargs)
        
    def _process_train_dir(self, txt_file, img_fold):
        dataset = []
        with open(osp.join(self.dataset_dir, txt_file),'r') as f:
           lines = f.readlines()
           for line in lines:
              line = line[:-1]
              img_path, pid, model, color = line.split(' ')
              img_path = osp.join(self.dataset_dir, img_fold+'/'+ img_path)
              
              pid = self.dataset_name + "_" + pid
              camid = self.dataset_name + "_0" 
              dataset.append((img_path+'.jpg', pid, camid))
 
        return dataset
        
    def _process_test_dir(self, txt_file, img_fold):
        dataset = []
        with open(osp.join(self.dataset_dir, txt_file),'r') as f:
           lines = f.readlines()
           for line in lines:
              line = line[:-1]
              img_path, pid, model, color = line.split(' ')
              img_path = osp.join(self.dataset_dir, img_fold+'/'+img_path+'.jpg') 
              dataset.append((img_path, int(pid), self.test_count))
              
              self.test_count += 1
        return dataset
        
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
