#encoding=utf-8
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from bbox_dataset import DetectionDataset
from lib.utils.common import lsdir
# from utils.common import lsdir
import json,os
import numpy as np
from PIL import Image
import mxnet as mx
from gluoncv.utils.viz.image import plot_image
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pprint
import pandas as pd
def dispFonts():
    #显示可用的中文字体，同时支持英文的
    from matplotlib.font_manager import FontManager
    import subprocess

    fm = FontManager()
    mat_fonts = set(f.name for f in fm.ttflist)

    output = subprocess.check_output(
        'fc-list :lang=zh -f "%{family}\n"', shell=True)
    output = output.decode('utf-8')
    # print '*' * 10, '系统可用的中文字体', '*' * 10
    # print output
    zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n'))
    available = mat_fonts & zh_fonts

    print '*' * 10 +  u'可用的中文字体'+'*' * 10
    for f in available:
        print(f)
from matplotlib.font_manager import FontProperties
myfont =  FontProperties(fname='/root/project/YaHei.Consolas.1.12.ttf',size=20)
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

def plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, ax=None,
              reverse_rgb=False, absolute_coordinates=True):
    """Visualize bounding boxes.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituded.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).

    Returns
    -------
    matplotlib axes
        The ploted axes.

    """
    from matplotlib import pyplot as plt

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if len(bboxes) < 1:
        return ax

    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=colors[cls_id],
                             linewidth=3.5)
        ax.add_patch(rect)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = u'{:.3f}'.format(scores.flat[i]) if scores is not None else ''
        if class_name or score:
            ax.text(xmin, ymin - 2,
                    u'{:s} {:s}'.format(class_name, score),
                    bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                    fontsize=12, color='white',fontproperties=myfont)
    return ax
class AluminumDet(DetectionDataset):
    def __init__(self, root = u"/data1/bupi_data/guangdong1_round1_train1_20190818/defect_Images/", is_train = True):
        super(AluminumDet,self).__init__()
        # self.classes = {
        #     '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
        #     '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
        #     '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
        #     '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
        # }
        self.classes = {
            u'\u7834\u6d1e': 1, u'\u6c34\u6e0d': 2, u'\u6cb9\u6e0d': 2, u'\u6c61\u6e0d': 2, u'\u4e09\u4e1d': 3, u'\u7ed3\u5934': 4,
            u'\u82b1\u677f\u8df3': 5, u'\u767e\u811a': 6, u'\u6bdb\u7c92': 7, u'\u7c97\u7ecf': 8, u'\u677e\u7ecf': 9, u'\u65ad\u7ecf': 10,
            u'\u540a\u7ecf': 11, u'\u7c97\u7ef4': 12, u'\u7eac\u7f29': 13, u'\u6d46\u6591': 14, u'\u6574\u7ecf\u7ed3': 15,
            u'\u661f\u8df3': 16, u'\u8df3\u82b1': 16, u'\u65ad\u6c28\u7eb6': 17, u'\u7a00\u5bc6\u6863': 18, u'\u6d6a\u7eb9\u6863': 18,
            u'\u8272\u5dee\u6863': 18, u'\u78e8\u75d5': 19, u'\u8f67\u75d5': 19, u'\u4fee\u75d5': 19, u'\u70e7\u6bdb\u75d5': 19,
            u'\u6b7b\u76b1': 20, u'\u4e91\u7ec7': 20, u'\u53cc\u7eac': 20, u'\u53cc\u7ecf': 20, u'\u8df3\u7eb1': 20,
            u'\u7b58\u8def': 20, u'\u7eac\u7eb1\u4e0d\u826f': 20,
        }
        anno_file_1 = "/data1/bupi_data/guangdong1_round1_train1_20190818/Annotations/anno_train.json"
        anno_file_2 = "/data1/bupi_data/guangdong1_round1_train2_20190828/Annotations/anno_train.json"

        anno_result_1 = pd.read_json(open(anno_file_1, "r"))
        anno_result_2 = pd.read_json(open(anno_file_2), 'r')

        self.objs = {}
        for ii in range(2):
            if ii==0:
                anno_result = anno_result_1
                image_dir = '/data1/bupi_data/guangdong1_round1_train1_20190818/defect_Images/'
            else:
                anno_result = anno_result_2
                image_dir = '/data1/bupi_data/guangdong1_round1_train2_20190828/defect_Images/'

            name_list = anno_result["name"].unique()
            image_list = list(lsdir(image_dir, suffix=u".jpg"))
            print(len(name_list))
            print(len(image_list))
            for img_name in name_list:
                img_annos = anno_result[anno_result["name"]==img_name]
                points = img_annos["bbox"].tolist()
                defect_names = img_annos["defect_name"].tolist()
                bboxes = []
                for i in range(np.shape(points)[0]):
                    x0, y0, x1, y1 = int(points[i][0]), int(points[i][1]), int(points[i][2]), int(points[i][3])
                    cls = self.classes[defect_names[i]]-1                     # waste so many time
                    if cls == 20:
                        print("cls = ", 20)
                    bboxes.append([x0, y0, x1, y1, cls])
                    # print([x0, y0, x1, y1, cls])
                filepath = os.path.join(image_dir, img_name)
                assert os.path.exists(filepath), pprint.pprint(filepath)
                self.objs[filepath] = bboxes
        self.names = list(self.objs.keys())
        self.names.sort()
        #     bboxes = []
        #     for bbox in anno["shapes"]:
        #         points = bbox['points']
        #         x0,y0,x1,y1 = points[0][0],points[0][1],points[1][0],points[3][1]
        #         cls = self.classes.index(bbox['label'])
        #         bboxes.append([x0,y0,x1,y1,cls])
        #     filepath = ann_file[:-5]+u".jpg"
        #     assert os.path.exists(filepath), pprint.pprint(filepath)
        #     self.objs[filepath] = bboxes
        # self.names = list(self.objs.keys())
        # self.names.sort()
        train_names, val_names = train_test_split(self.names, test_size=.1, random_state=42)
        is_train = False
        print('is_trian', is_train)
        if is_train:
            self.names = train_names
        else:
            self.names = val_names
    def at_with_image_path(self, idx):
        filepath = self.names[idx]
        return filepath, np.array(self.objs[filepath])
    def __len__(self):
        return len(self.names)
    def viz(self, indexes=None):
        import matplotlib.pyplot as plt
        if indexes is None:
            indexes = range(len(self))
        for index in indexes:
            x = self.at_with_image_path(index)
            plot_bbox(np.array(Image.open(x[0])), x[1][:, :4], labels=x[1][:, 4], class_names=self.classes)
            plt.show()

class AluminumDet_test(DetectionDataset):
    def __init__(self, root = u"/home/dell/DaringTang/data/guangdong_round2_train_20181011", is_train = True):
        super(AluminumDet_test,self).__init__()
        self.classes = [u"不导电", u"擦花", u"角位漏底", u"桔皮", u"漏底", u"喷流", u"漆泡", u'起坑', u'杂色', u'脏点']
        anno_list = list(lsdir(root, suffix=u".jpg"))
        print(len(anno_list))
        self.objs = {}
        for ann_file in anno_list:
            # anno = json.load(open(ann_file, "rb"))
            name = os.path.basename(ann_file)
            bboxes = []
            bboxes.append([25,25,250,250,1])
            filepath = ann_file
            assert os.path.exists(ann_file), pprint.pprint(filepath)
            self.objs[filepath] = bboxes
        self.names = list(self.objs.keys())
        self.names.sort()
    def at_with_image_path(self, idx):
        filepath = self.names[idx]
        return filepath, np.array(self.objs[filepath])
    def __len__(self):
        return len(self.names)
    def viz(self, indexes=None):
        import matplotlib.pyplot as plt
        if indexes is None:
            indexes = range(len(self))
        for index in indexes:
            x = self.at_with_image_path(index)

            plot_bbox(np.array(Image.open(x[0])), x[1][:, :4], labels=x[1][:, 4], class_names=self.classes)
            plt.show()
class AluminumDet_rotate(DetectionDataset):
    def __init__(self, root = u"/data3/zyx/project/al_detect/data/round2_train_rotate_img/train/", is_train = True):
        super(AluminumDet_rotate,self).__init__()
        self.classes = [u"不导电", u"擦花", u"角位漏底", u"桔皮", u"漏底", u"喷流", u"漆泡", u'起坑', u'杂色', u'脏点']
        json_data = json.load(open('/data3/zyx/project/al_detect/data/round2_train_rotate/AI_Lvcai_train.json'))
        anno_index = 0
        self.objs = {}
        for image in json_data[u'images']:
            id = image[u'id']
            img_name = image[u'file_name']
            bboxes = []
            while json_data[u'annotations'][anno_index][u'image_id']==id:
                this_anno = json_data[u'annotations'][anno_index]
                bboxes.append([this_anno[u'bbox'][0],
                               this_anno[u'bbox'][1],
                               this_anno[u'bbox'][0]+this_anno[u'bbox'][2],
                               this_anno[u'bbox'][1]+this_anno[u'bbox'][3],
                               this_anno[u'category_id']-1])
                anno_index+=1
                if anno_index==len(json_data[u'annotations']):
                    break
            self.objs[root+image[u'file_name']] = bboxes
        self.names = list(self.objs.keys())
        self.names.sort()
    def at_with_image_path(self, idx):
        filepath = self.names[idx]
        return filepath, np.array(self.objs[filepath])
    def __len__(self):
        return len(self.names)
    def viz(self, indexes=None):
        import matplotlib.pyplot as plt
        if indexes is None:
            indexes = range(len(self))
        for index in indexes:
            x = self.at_with_image_path(index)

            plot_bbox(np.array(Image.open(x[0])), x[1][:, :4], labels=x[1][:, 4], class_names=self.classes)
            plt.show()

if __name__ == '__main__':
    al = AluminumDet()#.viz()
    al.to_roidb("/home/dell/DaringTang/data/train_round12_val.roidb")                  # 生成roidb且其存放的路径