#encoding=utf-8
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from bbox_dataset import DetectionDataset
from lib.utils.common import lsdir
#from utils.common import lsdir
import json,os
import numpy as np
from PIL import Image
import mxnet as mx
from gluoncv.utils.viz.image import plot_image
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pprint
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
    def __init__(self, root = u"/media/dell/668e232d-4bee-4d33-8dd9-fe6b54b738d3/data/rssrai2019_object_detection/val", is_train = False):
        super(AluminumDet,self).__init__()
        self.classes = [u"large-vehicle", u"swimming-pool", u"helicopter", u"bridge", u"plane", u"ship",
                        u"soccer-ball-field", u'basketball-court', u'airport', u'container-crane', u"ground-track-field",
                        u"small-vehicle", u"harbor", u"baseball-diamond", u"tennis-court", u"roundabout",
                        u"storage-tank", u"helipad"]
        anno_list = list(lsdir(root, suffix=u".json"))
        self.objs = {}
        for ann_file in anno_list:
            anno = json.load(open(ann_file, "rb"))
            # name = anno["imagePath"]
            bboxes = []
            for bbox in anno["shapes"]:
                points = bbox['points']
                x0,y0,x1,y1 = points[0][0],points[0][1],points[1][0],points[1][1]
                cls = self.classes.index(bbox['label'])
                bboxes.append([x0,y0,x1,y1,cls])
            filepath = ann_file[:-5]+u".png"
            assert os.path.exists(filepath), pprint.pprint(filepath)
            self.objs[filepath] = bboxes
        self.names = list(self.objs.keys())
        self.names.sort()
        train_names, val_names = train_test_split(self.names, test_size=.1, random_state=42)
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
    def __init__(self, root = u"/media/dell/668e232d-4bee-4d33-8dd9-fe6b54b738d3/data/rssrai2019_object_detection/test", is_train = True):
        super(AluminumDet_test,self).__init__()
        self.classes = [u"large-vehicle", u"swimming-pool", u"helicopter", u"bridge", u"plane", u"ship",
                        u"soccer-ball-field", u'basketball-court', u'airport', u'container-crane', u"ground-track-field",
                        u"small-vehicle", u"harbor", u"baseball-diamond", u"tennis-court", u"roundabout",
                        u"storage-tank", u"helipad"]
        anno_list = list(lsdir(root, suffix=u".png"))
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
    Image.MAX_IMAGE_PIXELS = 1806504000
    al = AluminumDet_test() #.viz()
    al.to_roidb("/media/dell/668e232d-4bee-4d33-8dd9-fe6b54b738d3/data/rssrai2019_object_detection/test.roidb")             # 生成roidb且其存放的路径