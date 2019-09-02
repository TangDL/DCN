# coding=utf-8
import os
import codecs
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import shutil
import sys
from copy import deepcopy

import datetime
import re
import fnmatch
from PIL import Image


INFO = {
    "description": "lv_tianchi_dataset",
    "url": "https://cltdevelop@sina.cn",
    "version": "1.0",
    "year": 2018,
    "contributor": "cltdevelop",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "cltdevelop_licenses",
        "url": "http://cltdevelop@sina.cn/licenses"
    }
]

# defect_order = ['defect0', 'defect1', 'defect2', 'defect3',
#                 'defect4', 'defect5', 'defect6', 'defect7',
#                 'defect8', 'defect9', 'norm']
CATEGORIES = [
    {
        'id': 1,                 # note: id starts from 1
        'name': 'defect0',
        'supercategory': 'defection'
    },
    {
        'id': 2,
        'name': 'defect1',
        'supercategory': 'defection'
    },
    {
        'id': 3,
        'name': 'defect2',
        'supercategory': 'defection'
    },
    {
        'id': 4,
        'name': 'defect3',
        'supercategory': 'defection'
    },
    {
        'id': 5,
        'name': 'defect4',
        'supercategory': 'defection'
    },
    {
        'id': 6,
        'name': 'defect5',
        'supercategory': 'defection'
    },
    {
        'id': 7,
        'name': 'defect6',
        'supercategory': 'defection'
    },
    {
        'id': 8,
        'name': 'defect7',
        'supercategory': 'defection'
    },
    {
        'id': 9,
        'name': 'defect8',
        'supercategory': 'defection'
    },
    {
        'id': 10,
        'name': 'defect9',
        'supercategory': 'defection'
    },
]


def gen_coco_formart_dataset(path_root, path_json_list, save_root, dataset_name='train',
                             start_image_id=1, start_segmentation_id=1, list_aug=None):
    print('{} json files in {}'.format(len(path_json_list), path_root))

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    width_img = 2560
    height_img = 1920
    is_crowd = 0
    image_id = start_image_id
    segmentation_id = start_segmentation_id

    # go through each image/json
    cnt = 1
    for path_json in path_json_list:
        print('{} / {}'.format(cnt, len(path_json_list)))
        filename_without_root = os.path.basename(path_json)
        filename_without_root = filename_without_root.replace('.json', '')
        info_json = parse_one_json(path_json)
        labels = info_json['labels']
        points = info_json['points']
        path_img = os.path.join(path_root, dataset_name, filename_without_root + '.jpg')
        img = cv2.imread(path_img)

        for aug in list_aug:
            # ['vflip', 'noflip']
            points_tmp = deepcopy(points)
            points_tmp = [np.array(each) for each in points_tmp]
            if aug == 'hflip':
                # train_net_step.py already do hflip
                filename_without_root_tmp = filename_without_root + '_hflip' + '.jpg'
                for a_idx in range(len(points_tmp)):
                    points_tmp[a_idx][:, 0] = width_img - points_tmp[a_idx][:, 0]
                img_tmp = cv2.flip(img, 1)
                cv2.imwrite(os.path.join(path_root, dataset_name, filename_without_root_tmp), img_tmp)
            elif aug == 'vflip':
                filename_without_root_tmp = filename_without_root + '_vflip' + '.jpg'
                for a_idx in range(len(points_tmp)):
                    points_tmp[a_idx][:, 1] = height_img - points_tmp[a_idx][:, 1]
                img_tmp = cv2.flip(img, 0)
                cv2.imwrite(os.path.join(path_root, dataset_name, filename_without_root_tmp), img_tmp)
            elif aug == 'noflip':
                filename_without_root_tmp = filename_without_root + '.jpg'
                img_tmp = img
            elif aug == -7:
                filename_without_root_tmp = filename_without_root + '_minus_7.jpg'
                img_tmp, _, points_tmp = im_rotate_v2(img, points_tmp, aug)
                cv2.imwrite(os.path.join(path_root, dataset_name, filename_without_root_tmp), img_tmp)
            elif aug == 7:
                filename_without_root_tmp = filename_without_root + '_plus_7.jpg'
                img_tmp, _, points_tmp = im_rotate_v2(img, points_tmp, aug)
                cv2.imwrite(os.path.join(path_root, dataset_name, filename_without_root_tmp), img_tmp)
            else:
                print('wrong aug in gen_coco_formart_dataset')
                sys.exit(0)
            image_info = {
                'id': image_id,
                'file_name': filename_without_root_tmp,
                'width': width_img,
                'height': height_img,
                'date_captured': datetime.datetime.utcnow().isoformat(' '),
                'license': LICENSES[0]['id'],
                'coco_url': '',
                'flickr_url': ''
            }
            coco_output['images'].append(image_info)

            # for annotations/bbox
            for idx in range(len(labels)):
                label = labels[idx]
                bbox = np.array(points_tmp[idx])
                xmin, xmax = int(np.min(bbox[:, 0])), int(np.max(bbox[:, 0]))
                ymin, ymax = int(np.min(bbox[:, 1])), int(np.max(bbox[:, 1]))

                segmentation = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]

                class_id = [x['id'] for x in CATEGORIES if x['name'] == label][0]
                annotation_info = {
                    'id': segmentation_id,
                    'image_id': image_id,
                    'category_id': class_id,
                    'iscrowd': is_crowd,
                    'area': (xmax - xmin) * (ymax - ymin),
                    # x, y, width, height
                    'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                    'segmentation': [segmentation],

                }
                coco_output['annotations'].append(annotation_info)

                segmentation_id += 1
            image_id += 1
        cnt += 1
    save_path = os.path.join(save_root, 'AI_Lvcai_' + dataset_name + '.json')
    with open(save_path, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print('save {} over'.format(save_path))

    return image_id, segmentation_id
#########################################################################


chi_defect_code = {
    '不导电': 'defect0',
    '擦花': 'defect1',
    '角位漏底': 'defect2',
    '桔皮': 'defect3',
    '漏底': 'defect4',
    '喷流': 'defect5',
    '漆泡': 'defect6',
    '起坑': 'defect7',
    '杂色': 'defect8',
    '脏点': 'defect9',
    '正常': 'norm'
}

eng_defect_code = {
    'defect0': '不导电',
    'defect1': '擦花',
    'defect2': '角位漏底',
    'defect3': '桔皮',
    'defect4': '漏底',
    'defect5': '喷流',
    'defect6': '漆泡',
    'defect7': '起坑',
    'defect8': '杂色',
    'defect9': '脏点',
    'norm': '正常'
}


defect_order = ['defect0', 'defect1', 'defect2', 'defect3',
                'defect4', 'defect5', 'defect6', 'defect7',
                'defect8', 'defect9', 'norm']


def parse_one_json(path_json):
    info_json = {}
    info_json['points'] = []
    info_json['labels'] = []

    with codecs.open(path_json, encoding='utf-8') as f_json:
        contents = json.load(f_json)
    # print(contents)
    shapes = contents['shapes']

    for idx in range(len(shapes)):
        one_fault = shapes[idx]
        # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        points = one_fault['points']
        label = one_fault['label']
        label = chi_defect_code[label]  # defect0, defect1,...

        info_json['points'].append(points)
        info_json['labels'].append(label)

    return info_json


def vis_one_sample(path_img, info_json):
    img = cv2.imread(path_img)
    bboxes = info_json['points']
    labels = info_json['labels']
    for idx in range(len(labels)):
        bbox = np.array(bboxes[idx])
        label = labels[idx]
        xmin = int(np.min(bbox[:, 0]))
        xmax = int(np.max(bbox[:, 0]))
        ymin = int(np.min(bbox[:, 1]))
        ymax = int(np.max(bbox[:, 1]))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)

    plt.figure()
    plt.imshow(img[:, :, ::-1])
    plt.show()


def split_train_val_json(path_root, ratio=0.1, random_seed=27):
    path_json_list = glob.glob(os.path.join(path_root, '*/*/*.json'))
    path_json_list += glob.glob(os.path.join(path_root, '*/*.json'))
    print('{} json files in {}'.format(len(path_json_list), path_root))

    # get label for each json
    labels_list = []
    for path_json in path_json_list:
        info_json = parse_one_json(path_json)
        tgt_label = list(set(info_json['labels']))
        tgt_label = ''.join(tgt_label)
        labels_list.append(tgt_label)

    # split train and val by category
    category = {}
    category['multi_defect'] = []
    for idx in range(len(labels_list)):
        cls = labels_list[idx]
        if len(cls) <= 7:
            # single defection or norm
            if cls not in category.keys():
                category[cls] = []
            category[cls].append(path_json_list[idx])
        else:
            category['multi_defect'].append(path_json_list[idx])
    for defect in category.keys():
        print('{}, {} samples'.format(defect, len(category[defect])))

    val = []
    train = []
    random.seed(random_seed)
    for key in category.keys():
        one_class = category[key]
        length_val = int(len(one_class) * ratio)
        length_train = len(one_class) - length_val
        random.shuffle(one_class)
        part_val = [one_class[i] for i in range(length_val)]
        part_train = list(set(one_class) - set(part_val))
        val.extend(part_val)
        train.extend(part_train)
    print('train samples: {}, val samples: {}'.format(len(train), len(val)))

    # copy images
    if not os.path.exists(os.path.join(path_root, 'train')):
        os.makedirs(os.path.join(path_root, 'train'))
    if not os.path.exists(os.path.join(path_root, 'val')):
        os.makedirs(os.path.join(path_root, 'val'))
    for path_json in train:
        path_img = path_json.replace('.json', '.jpg')
        filename = os.path.basename(path_img)
        save_path_img = os.path.join(path_root, 'train', filename)
        shutil.copyfile(path_img, save_path_img)

    # rename filenames in val for both image file and json file
    # convenient to pred_val.py
    val = sorted(val)
    for idx in range(len(val)):
        path_json = val[idx]
        path_img = path_json.replace('.json', '.jpg')
        new_filename = str(idx + 1)
        # filename = os.path.basename(path_img)
        save_path_img = os.path.join(path_root, 'val', new_filename + '.jpg')
        save_path_json = os.path.join(path_root, 'val', new_filename + '.json')
        shutil.copyfile(path_img, save_path_img)
        shutil.copy(path_json, save_path_json)

        val[idx] = save_path_json

    return train, val


def im_rotate_v2(cv_img, keypoints_list, angle):
    """
    :param cv_img:
    :param keypoints_list:
    :param limb_points:
    :return:
    """
    w, h = cv_img.shape[1], cv_img.shape[0]
    matrix = cv2.getRotationMatrix2D((w // 2.0, h // 2.0), angle, 1.0)
    rotation_img = cv2.warpAffine(cv_img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    new_keypoints = []
    new_all_bbox = []

    for idx in range(len(keypoints_list)):
        bbox = np.array(keypoints_list[idx])
        new_bbox = []
        for i_pt in range(bbox.shape[0]):
            x, y = bbox[i_pt][0], bbox[i_pt][1]
            p0 = np.array([x, y]).astype(np.float32).reshape(1, 1, 2)
            p = cv2.transform(p0, matrix).reshape(1, -1)
            x = p[0, 0]
            y = p[0, 1]
            new_bbox.append([x, y])

        # convert new_bbox to rectangle
        new_x, new_y, new_w, new_h = cv2.boundingRect(np.array(new_bbox))
        xmin, ymin, xmax, ymax = new_x, new_y, new_x + new_w, new_y + new_h
        xmin = np.max([0, xmin])
        ymin = np.max([0, ymin])
        xmax = np.min([xmax, w])
        ymax = np.min([ymax, h])

        new_all_bbox.append(np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]))
        new_keypoints.append(np.array(new_bbox))

    return rotation_img, new_keypoints, new_all_bbox


if __name__ == '__main__':
    path_root = '/data1/zyx/cltdevelop/lvcai/guangdong_round2_train_20181011'
    save_root = '/data3/zyx/project/al_detect/data/round2_train_rotate'

    # data augmentation
    aug_list = ['vflip', 'noflip']
    rotation_angle_list = [-7, 7]

    train_json_list, val_json_list = split_train_val_json(path_root, ratio=0.1, random_seed=27)
    image_id, segmentation_id = gen_coco_formart_dataset(
        path_root, train_json_list, save_root, dataset_name='train',
        start_image_id=1, start_segmentation_id=1, list_aug=['vflip', 'noflip', -7, 7])
    gen_coco_formart_dataset(path_root, val_json_list, save_root, dataset_name='val',
                             start_image_id=image_id, start_segmentation_id=segmentation_id,
                             list_aug=['noflip'])



