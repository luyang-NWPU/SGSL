# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import json
import os.path as osp
from .bases import read_json, write_json
import pdb
from fastreid.utils.file_io import PathManager
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class CUHK01(ImageDataset):
    dataset_dir = 'cuhk01'
    dataset_url = None
    dataset_name = "CUHK01"

    def __init__(self, root='datasets', split_id=0, CUHK01_labeled=True, CUHK01_classic_split=False, **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.zip_path = osp.join(self.dataset_dir, 'CUHK01.zip')
        self.campus_dir = osp.join(self.dataset_dir, 'campus')
        self.split_path = osp.join(self.dataset_dir, 'splits.json')

        self.extract_file()

        required_files = [self.dataset_dir, self.campus_dir]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'
                .format(split_id,
                        len(splits) - 1)
            )
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']
        
        train = [(item[0], self.dataset_name+'_'+str(item[1]) , item[2]) for item in train]

        super(CUHK01, self).__init__(train, query, gallery, **kwargs)

    def extract_file(self):
        if not osp.exists(self.campus_dir):
            print('Extracting files')
            zip_ref = zipfile.ZipFile(self.zip_path, 'r')
            zip_ref.extractall(self.dataset_dir)
            zip_ref.close()

    def prepare_split(self):
        """
        Image name format: 0001001.png, where first four digits represent identity
        and last four digits represent cameras. Camera 1&2 are considered the same
        view and camera 3&4 are considered the same view.
        """
        if not osp.exists(self.split_path):
            print('Creating 10 random splits of train ids and test ids')
            img_paths = sorted(glob.glob(osp.join(self.campus_dir, '*.png')))
            img_list = []
            pid_container = set()
            for img_path in img_paths:
                img_name = osp.basename(img_path)
                pid = int(img_name[:4]) - 1
                camid = (int(img_name[4:7]) - 1) // 2 # result is either 0 or 1
                img_list.append((img_path, pid, camid))
                pid_container.add(pid)

            num_pids = len(pid_container)
            num_train_pids = num_pids // 2

            splits = []
            for _ in range(10):
                order = np.arange(num_pids)
                np.random.shuffle(order)
                train_idxs = order[:num_train_pids]
                train_idxs = np.sort(train_idxs)
                idx2label = {
                    idx: label
                    for label, idx in enumerate(train_idxs)
                }

                train, test_a, test_b = [], [], []
                for img_path, pid, camid in img_list:
                    if pid in train_idxs:
                        train.append((img_path, idx2label[pid], camid))
                    else:
                        if camid == 0:
                            test_a.append((img_path, pid, camid))
                        else:
                            test_b.append((img_path, pid, camid))

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
