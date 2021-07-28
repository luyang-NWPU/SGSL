# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob, pdb
import os
import os.path as osp
from collections import OrderedDict, defaultdict
import re
import warnings
from scipy.io import loadmat
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
import numpy as np
from tqdm import tqdm
from PIL import Image
import copy

@DATASET_REGISTRY.register()
class cuhkSYSU(ImageDataset):
    """CUHK SYSU datasets.

    The dataset is collected from two sources: street snap and movie.
    In street snap, 12,490 images and 6,057 query persons were collected
    with movable cameras across hundreds of scenes while 5,694 images and
    2,375 query persons were selected from movies and TV dramas.

    Dataset statistics:
        - identities: xxx.
        - images: 12936 (train).
    """
    dataset_name = 'cuhk-sysu'
    #+--------------------------------+--------+------------+---------+
    #|              set               | images | identities | cameras |
    #+--------------------------------+--------+------------+---------+
    #| IncrementalSamples4subcuhksysu |        |            |         |
    #|             train              |  4374  |    942     |    1    |
    #|             query              |  2900  |    2900    |    1    |
    #|            gallery             |  5447  |    2900    |    1    |
    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_name, 'CUHK-SYSU')
        self.data_path = osp.join(self.dataset_dir, 'Image', 'SSM')
        self.annotation_path = osp.join(self.dataset_dir, 'annotation')
        
        self.processed_dir = osp.join(self.dataset_dir, 'cuhksysu4reid')
        self.processed_dir_train = osp.join(self.processed_dir, 'train')
        self.processed_dir_query = osp.join(self.processed_dir, 'query')
        self.processed_dir_gallery = osp.join(self.processed_dir, 'gallery')
        self.processed_dir_combine = osp.join(self.processed_dir, 'combine')
        required_files_state = [self.processed_dir_train, self.processed_dir_query, self.processed_dir_gallery,
                                self.processed_dir_combine]
        self.cam_index = 0
        
        if all(map(osp.exists, required_files_state)):
            train = self.process_dir(self.processed_dir_train, isTrain=True)
            _combine = self.process_dir(self.processed_dir_combine, isTrain=True)
            query = self.process_dir(self.processed_dir_query, isTrain=False)
            gallery = self.process_dir(self.processed_dir_gallery, isTrain=False)
        else:
            if osp.exists(self.processed_dir) is False:
                os.mkdir(self.processed_dir)
            os.mkdir(self.processed_dir_train)
            os.mkdir(self.processed_dir_combine)
            os.mkdir(self.processed_dir_query)
            os.mkdir(self.processed_dir_gallery)

            self.preprocessing()
            train = self.process_dir(self.processed_dir_train, isTrain=True)
            _combine = self.process_dir(self.processed_dir_combine, isTrain=True)
            query = self.process_dir(self.processed_dir_query, isTrain=False)
            gallery = self.process_dir(self.processed_dir_gallery, isTrain=False)

        self.train, self.query, self.gallery = train, query, gallery
        
        self.sub_set()
        
        super(cuhkSYSU, self).__init__(self.train, self.query, self.gallery, **kwargs)
        
    def sub_set(self):
        print('### Using subset of CUHK-SYSU ###')
        results, bigger4_list, sub_train = {}, [], []
        for it in self.train:
            if it[1] not in results.keys():
                results[it[1]] = 1
            else:
                results[it[1]] += 1
        for key, value in results.items():
            if value >= 4:
                bigger4_list.append(key)
        for it in self.train:
            if it[1] in bigger4_list:
                sub_train.append(it)
        
        self.train = sub_train
        
        #print('### image: %d, ID: %d ###'%(len(self.train),len(bigger4_list)))
        
    def process_dir(self, dir_path, isTrain=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_s([-\d]+)_([-\d]+)_([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, image_name, bbox_index, is_hard = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, _, _, _ = map(int, pattern.search(img_path).groups())
            self.cam_index += 1
            camid = self.cam_index
            if isTrain:
                pid = pid2label[pid]
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
                data.append((img_path, pid, camid))
            else:
                data.append((img_path, pid, camid))

        return data
        
    def crop_store(self, data_dict, save_dir):
        def _crop_store():
            name = save_dir.split('/')[-1]
            image_dict = defaultdict(list)

            index_instance = 0
            for key, person_images in tqdm(data_dict.items()):
                for image_path, box, pid_name, pid, im_name, is_hard in person_images:
                    assert osp.exists(image_path)
                    one_img = Image.open(image_path)
                    one_img_copy = copy.deepcopy(one_img)
                    box_tuple = tuple(box.round())
                    box_tuple = map(int, box_tuple)
                    filled_pid = str(pid).zfill(5)
                    is_hard = str(is_hard)
                    cropped = one_img_copy.crop(box_tuple)
                    image_name = im_name.replace('.jpg', '')
                    cropped_path = osp.join(save_dir,
                                            f'{filled_pid}_{image_name}_{str(index_instance).zfill(7)}_{is_hard}.jpg')
                    cropped.save(cropped_path)
                    image_dict[pid_name].append((cropped_path, int(pid), 0, 'cuhksysu', int(pid)))
                    index_instance = index_instance + 1

            print(f'Finished processing {name} dir!')
            return image_dict

        if osp.exists(save_dir) is False:
            os.makedirs(save_dir)
            _crop_store()
        else:
            _crop_store()

    def preprocessing(self):
        Train_mat = loadmat(osp.join(self.annotation_path, 'test', 'train_test', 'Train.mat'))
        testg50_mat = loadmat(osp.join(self.annotation_path, 'test', 'train_test', 'TestG50.mat'))['TestG50'].squeeze()
        all_imgs_mat = loadmat(osp.join(self.annotation_path, 'Person.mat'))
        # pool_mat = loadmat(osp.join(self.annotation_path, 'pool.mat'))
        id_name_to_pid = {}
        train_pid_dict = defaultdict(list)

        train = Train_mat['Train'].squeeze()
        n_train = 0
        for index, item in enumerate(train):
            pid_name = item[0, 0][0][0]
            pid = int(pid_name[1:])
            id_name_to_pid[pid_name] = pid
            scenes = item[0, 0][2].squeeze()
            for im_name, box, is_hard in scenes:
                im_name = str(im_name[0])
                is_hard = is_hard[0][0]
                box = box.squeeze().astype(np.int32)
                box[2:] += box[:2]
                image_path = osp.join(self.data_path, im_name)
                train_pid_dict[pid_name].append((image_path, box, pid_name, pid, im_name, is_hard))
                n_train = n_train + 1

        probe_pid_dict = defaultdict(list)
        gallery_pid_dict = defaultdict(list)
        n_probe = 0
        n_gallery = 0
        for query, gallery in zip(testg50_mat['Query'], testg50_mat['Gallery']):
            im_name = str(query['imname'][0, 0][0])
            roi = query['idlocate'][0, 0][0].astype(np.int32)
            roi[2:] += roi[:2]
            is_hard = query['ishard'][0, 0][0, 0]
            pid_name = query['idname'][0, 0][0]
            pid = int(pid_name[1:])
            assert pid_name not in id_name_to_pid.keys()
            id_name_to_pid[pid_name] = pid
            # im_name, bbox, is_hard, idname, flipped
            image_path = osp.join(self.data_path, im_name)
            probe_pid_dict[pid_name].append((image_path, roi, pid_name, pid, im_name, is_hard))
            n_probe = n_probe + 1
            gallery = gallery.squeeze()
            for _gallery in gallery:
                _im_name = str(_gallery['imname'][0])
                _roi = _gallery['idlocate'][0].astype(np.int32)
                if _roi.size == 0:
                    continue
                else:
                    _roi[2:] += _roi[:2]
                    _is_hard = _gallery['ishard'][0][0]
                    # _id_name = _gallery['idname'][0]
                    # im_name, bbox, is_hard, idname, flipped
                    _image_path = osp.join(self.data_path, _im_name)
                    gallery_pid_dict[pid_name].append((_image_path, _roi, pid_name, pid, _im_name, _is_hard))
                    n_gallery = n_gallery + 1

        num_total_pid = len(train_pid_dict) + len(probe_pid_dict)

        print(num_total_pid)
        all_image_dict = defaultdict(list)
        all_imgs = all_imgs_mat['Person'].squeeze()

        n = 0
        for id_name, _, scenes in all_imgs:
            pid_name = id_name[0]
            pid = int(pid_name[1:])
            scenes = scenes.squeeze()
            for im_name, box, is_hard in scenes:
                im_name = str(im_name[0])
                is_hard = is_hard[0, 0]
                box = box.squeeze().astype(np.int32)
                box[2:] += box[:2]
                image_path = osp.join(self.data_path, im_name)
                all_image_dict[pid_name].append((image_path, box, pid_name, pid, im_name, is_hard))
                n = n + 1

        print(n)
        print(f'n_train: {n_train}, n_probe: {n_probe}, n_gallery: {n_gallery} n_all:{n}')
        train_dict = self.crop_store(train_pid_dict, osp.join(self.processed_dir, 'train'))
        probe_dict = self.crop_store(probe_pid_dict, osp.join(self.processed_dir, 'query'))
        gallery_dict = self.crop_store(gallery_pid_dict, osp.join(self.processed_dir, 'gallery'))
        all_dict = self.crop_store(all_image_dict, osp.join(self.processed_dir, 'combine'))
        