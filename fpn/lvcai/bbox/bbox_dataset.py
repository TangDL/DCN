import datetime
import json
import os

import cv2
import mxnet.ndarray as nd
import numpy as np
from mxnet.gluon.data import Dataset
from tqdm import tqdm
import mxnet as mx
from PIL import Image

class DetectionDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super(DetectionDataset, self).__init__()
        self.transforms = None
    def __getitem__(self, idx):
        img_path,bbox = self.at_with_image_path(idx)
        img = cv2.imread(img_path)[:,:,::-1]
        return nd.array(img),nd.array(bbox)
    def __len__(self):
        return 0
    def viz(self, indexes=None):
        from gluoncv.utils.viz import plot_bbox
        import matplotlib.pyplot as plt
        if indexes is None:
            indexes = range(len(self))
        for index in indexes:
            x = self.at_with_image_path(index)
            img_path = x[0]
            img = cv2.imread(img_path)[:,:,::-1]
            plot_bbox(img, x[1][:, :4], labels=x[1][:, 4], class_names=self.classes)
            plt.show()
    def viz_seg(self, indexes=None):
        from gluoncv.utils.viz import plot_bbox
        import matplotlib.pyplot as plt
        if indexes is None:
            indexes = range(len(self))
        for index in indexes:
            x = self.at_with_image_path(index)
            img_path = x[0]
            img = cv2.imread(img_path)[:,:,::-1]
            plot_bbox(img, x[1][:, :4], labels=x[1][:, 4], class_names=self.classes)
            for k in range(len(x[2])):
                p = plt.Polygon(zip(x[2][k][0][:,0],x[2][k][0][:,1]),alpha = .4)
                plt.gca().add_patch(p)
            plt.show()

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_class(self):
        return self.num_classes

    def at_with_image_path(self, idx):
        '''
        return img_path and bbox, implement this if you want to use to_coco.
        Note the bbox in this case cannot be transformed.
        '''
        raise NotImplementedError()

    def to_roidb(self, db_path):
        '''
        Compressing this dataset to a pickeled roidb format file, which may
        be helpful if you want to train your own Deformable Convnets Model.
        '''
        r = []
        for i in tqdm(range(len(self))):
            img_path_ori, bbox = self.at_with_image_path(i)
            bbox = np.array(bbox).astype(np.float32)
            image = Image.open(img_path_ori)
            width, height = image.size
            # img = cv2.imread(img_path_ori)

            onedb = {}
            onedb["boxes"] = bbox[:, :4].astype(np.int32)
            onedb["height"] = height
            onedb["width"] = width
            onedb["image"] = img_path_ori
            onedb["flipped"] = False

            num_objs = bbox.shape[0]
            assert num_objs > 0
            num_classes = self.num_classes + 1
            overlaps = np.zeros(shape=(num_objs, num_classes), dtype=np.float32)
            for idx in range(bbox.shape[0]):
                cls = bbox[idx, 4]
                overlaps[idx, int(cls)] = 1.0
            onedb["gt_classes"] = bbox[:, 4].astype(np.int32) + 1
            onedb["gt_overlaps"] = overlaps
            onedb["max_classes"] = overlaps.argmax(axis=1)
            onedb["max_overlaps"] = overlaps.max(axis=1)
            r.append(onedb)
        import pickle
        pickle.dump(r, open(db_path, "wb"), protocol=0)
        return r

    def pascal_write_box(self, xml_path, boxes, filepath):
        width,height = Image.open(filepath).size
        objs = ""
        for box in boxes[:,:5]:
            x0, y0, x1, y1, cls = box
            class_name = self.classes[int(cls)]
            one_obj = '''
            <object><name>{0}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>
            <bndbox>
                <xmin>{1}</xmin>
                <ymin>{2}</ymin>
                <xmax>{3}</xmax>
                <ymax>{4}</ymax>
            </bndbox>
            </object>
            '''.format(class_name,x0,y0,x1,y1)
            objs += one_obj
        str2write = '''
        <annotation>
            <folder>{}</folder>
            <filename>{}</filename>
            <path>{}</path>
            <source>
            <database>Unknown</database>
            </source>
            <size>
                <width>{}</width>
                <height>{}</height>
                <depth>3</depth>
            </size>
            <segmented>0</segmented>
            {}
        </annotation>

        '''.format(filepath.strip().split('/')[-2],
                   os.path.basename(filepath),
                   filepath,
                   width,height,objs)
        print xml_path
        with open(xml_path,"wt") as f:
            f.write(str2write)
    def to_pascal(self, dir):
        for i in tqdm(range(len(self))):
            filename,bboxes = self.at_with_image_path(i)
            xml_path = os.path.join(dir, os.path.splitext(os.path.basename(filename))[0] + ".xml")
            self.pascal_write_box(xml_path,bboxes, filename)

    def parser_pascal_voc_xml(self, xml_path, img_root):
        import xml.etree.ElementTree as ET
        import logging
        oneimg = {}
        oneimg['bndbox'] = []
        try:
            dom = ET.parse(xml_path)
        except Exception as e:
            logging.error("{}_{}".format(e, xml_path))
            return None
        root = dom.getroot()
        filename = root.findall('filename')[0].text
        oneimg['path'] = os.path.join(img_root, filename)
        oneimg['filename'] = filename
        for objects in root.findall('object'):
            name = objects.find('name').text
            points = list(objects.find('bndbox'))
            if len(points) != 4:
                logging.warning("find illegal label in file:{}.xml. ".format(filename))
                print(points)
                return None
            xmin = int(points[0].text)
            ymin = int(points[1].text)
            xmax = int(points[2].text)
            ymax = int(points[3].text)

            oneimg['bndbox'].append([xmin, ymin, xmax, ymax, name])
        return oneimg