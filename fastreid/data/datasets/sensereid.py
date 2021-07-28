# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

import os.path as osp
from .bases import read_json, write_json
__all__ = ['SenseReID', ]


@DATASET_REGISTRY.register()
class SenseReID(ImageDataset):
    """Sense reid
    """
    dataset_dir = "sensereid"
    dataset_name = "sensereid"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'SenseReID', 'test_probe')
        self.gallery_dir = osp.join(
            self.dataset_dir, 'SenseReID', 'test_gallery'
        )

        required_files = [self.dataset_dir, self.query_dir, self.gallery_dir]
        self.check_before_run(required_files)

        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        # relabel
        g_pids = set()
        for _, pid, _ in gallery:
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        query = [
            (img_path, pid2label[pid], camid, 'sensereid', pid2label[pid])
            for img_path, pid, camid in query
        ]
        gallery = [
            (img_path, pid2label[pid], camid, 'sensereid', pid2label[pid])
            for img_path, pid, camid in gallery
        ]
        train = [] #copy.deepcopy(query) + copy.deepcopy(gallery) # dummy variable

        super(SenseReID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        data = []

        for img_path in img_paths:
            img_name = osp.splitext(osp.basename(img_path))[0]
            pid, camid = img_name.split('_')
            pid, camid = int(pid), int(camid)
            data.append((img_path, pid, camid))

        return data

