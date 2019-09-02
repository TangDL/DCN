# encoding=utf-8
# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

import numpy as np
from dataset import *

import pickle
import copy
import logging

def append_flipped_images(roidb):
    """
    append flipped images to an roidb
    flip boxes coordinates, images will be actually flipped when loading into network
    :param roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    """
    logging.info('append flipped images to roidb')
    num_images = len(roidb)
    roidb_r = copy.deepcopy(roidb)
    for i in range(num_images):
        roi_rec = roidb[i]
        boxes = roi_rec['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = roi_rec['width'] - oldx2 - 1
        boxes[:, 2] = roi_rec['width'] - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        entry = {'image': roi_rec['image'],
                 'height': roi_rec['height'],
                 'width': roi_rec['width'],
                 'boxes': boxes,
                 'gt_classes': roidb[i]['gt_classes'],
                 'gt_overlaps': roidb[i]['gt_overlaps'],
                 'max_classes': roidb[i]['max_classes'],
                 'max_overlaps': roidb[i]['max_overlaps'],
                 'flipped': True}

        roidb_r.append(entry)
    return roidb_r

def vappend_flipped_images(roidb):
    """
    append flipped images to an roidb
    flip boxes coordinates, images will be actually flipped when loading into network
    :param roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    """
    logging.info('append flipped images to roidb')
    num_images = len(roidb)
    roidb_r = copy.deepcopy(roidb)
    for i in range(num_images):
        roi_rec = roidb[i]
        boxes = roi_rec['boxes'].copy()
        oldy1 = boxes[:, 1].copy()
        oldy2 = boxes[:, 3].copy()
        boxes[:, 1] = roi_rec['height'] - oldy2 - 1
        boxes[:, 3] = roi_rec['height'] - oldy1 - 1
        assert (boxes[:, 3] >= boxes[:, 1]).all()
        entry = {'image': roi_rec['image'],
                 'height': roi_rec['height'],
                 'width': roi_rec['width'],
                 'boxes': boxes,
                 'gt_classes': roidb[i]['gt_classes'],
                 'gt_overlaps': roidb[i]['gt_overlaps'],
                 'max_classes': roidb[i]['max_classes'],
                 'max_overlaps': roidb[i]['max_overlaps'],
                 'vflipped': True,
                 "flipped": roidb[i]['flipped'] if 'flipped' in roidb[i].keys() else False
                 }

        roidb_r.append(entry)
    return roidb_r

def load_gt_roidb(dataset_name, image_set_name, root_path, dataset_path, result_path=None,
                  flip=False):
    roidb_dir="/home/dell/DaringTang/data/train_round2.roidb"
    with open(roidb_dir,"rb") as f:
        roidb = pickle.load(f)                                               # 使用pickle中的load模块反序列化，将其加载为一个python对象
    roidb = append_flipped_images(roidb)                                     # 加入翻转过后的标注信息
    roidb = vappend_flipped_images(roidb)
    return roidb
def load_proposal_roidb(dataset_name, image_set_name, root_path, dataset_path, result_path=None,
                        proposal='rpn', append_gt=True, flip=False):
    """ load proposal roidb (append_gt when training) """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)

    gt_roidb = imdb.gt_roidb()
    roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb, append_gt)
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def merge_roidb(roidbs):
    """ roidb are list, concat them together """
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    return roidb


def filter_roidb(roidb, config):
    """ remove roidb entries without usable rois """

    def is_valid(entry):
        """ valid images have at least 1 fg or bg roi """
        overlaps = entry['max_overlaps']
        fg_inds = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
        bg_inds = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'filtered %d roidb entries: %d -> %d' % (num - num_after, num, num_after)

    return filtered_roidb


def load_gt_segdb(dataset_name, image_set_name, root_path, dataset_path, result_path=None,
                  flip=False):
    """ load ground truth segdb """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)
    segdb = imdb.gt_segdb()
    if flip:
        segdb = imdb.append_flipped_images_for_segmentation(segdb)
    return segdb


def merge_segdb(segdbs):
    """ segdb are list, concat them together """
    segdb = segdbs[0]
    for r in segdbs[1:]:
        segdb.extend(r)
    return segdb
