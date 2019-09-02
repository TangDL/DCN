# encoding=utf-8
from __future__ import print_function
import sys
sys.path.append('/home/dell/DaringTang/works/Detection/code/DCN/lib')
sys.path.append("..")
import cv2
import mxnet as mx
import mxnet.gluon.nn as nn
import mxnet.ndarray as nd
import numpy as np
from lib.bbox.bbox_transform import bbox_pred, clip_boxes
from lib.nms.nms import gpu_nms_wrapper
from lib.utils.image import transform
from lib.utils.common import lsdir

from config.config import config, update_config
from symbols import *
import argparse

import sys

# import os
# os.environ['CUDA_VISIBLE_DEVICES']='2,3'              # 指定使用的GPU

reload(sys)
sys.setdefaultencoding('utf-8')

def load_checkpoint(prefix):
    save_dict = mx.nd.load(prefix)
    arg_params = {}
    aux_params = {}

    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def load_param(prefix, convert=False, ctx=None, process=True):
    arg_params, aux_params = load_checkpoint(prefix)
    if process:
        tests = [k for k in arg_params.keys() if '_test' in k]
        for test in tests:
            arg_params[test.replace('_test', '')] = arg_params.pop(test)
    return arg_params, aux_params


class SymbolBlock(nn.Block):
    def __init__(self, sym, input_names=("data",), resume=None, pretrained=None):
        super(SymbolBlock, self).__init__()
        self.features = mx.gluon.nn.SymbolBlock(
            sym,
            [mx.sym.Variable(x) for x in input_names]
        )

        net_params = self.features.collect_params()
        if resume is not None:
            args, auxes = load_param(resume)
            for param in args.keys():
                if param in net_params:
                    net_params[param]._load_init(args[param], ctx=mx.cpu())
            for param in auxes.keys():
                if param in net_params:
                    net_params[param]._load_init(auxes[param], ctx=mx.cpu())

        if pretrained is not None:
            args = mx.nd.load(pretrained)
            for param in args.keys():
                net_params = self.features.collect_params()
                if param in net_params:
                    net_params[param]._load_init(args[param], ctx=mx.cpu())

    def forward(self, *args):
        return self.features(*args)


def get_symbol(cfg):
    sym = resnet_v1_101_fpn_dcn_rcnn.resnet_v1_101_fpn_dcn_rcnn().get_symbol(cfg, is_train=False)
    return sym



def im_detect_bbox_aug(net, nms_wrapper, img_path, scales, pixel_means,
                       bbox_stds, ctx, threshold=1e-1, viz=False):
    all_bboxes = []
    all_scores = []
    img_ori = cv2.imread(img_path.encode("utf-8"))
    for scale_min, scale_max in scales:
        fscale = 1.0 * scale_min / min(img_ori.shape[:2])
        img_resized = cv2.resize(img_ori, (0, 0), fx=fscale, fy=fscale)
        h, w, c = img_resized.shape
        h_padded = h if h % 32 == 0 else h + 32 - h % 32
        w_padded = w if w % 32 == 0 else w + 32 - w % 32
        img_padded = np.zeros(shape=(h_padded, w_padded, c), dtype=img_resized.dtype)
        img_padded[:h, :w, :] = img_resized
        img = transform(img_padded, pixel_means=pixel_means)
        im_info = nd.array([[h_padded, w_padded, 1.0]], ctx=ctx[0])
        data = nd.array(img, ctx=ctx[0])

        rois, scores, bbox_deltas = net(data, im_info)
        rois = rois[:, 1:].asnumpy()
        bbox_deltas = bbox_deltas[0].asnumpy()
        # bbox_deltas = pre_compute_deltas(bbox_deltas, bbox_stds=bbox_stds)
        bbox = bbox_pred(rois, bbox_deltas)
        bbox = clip_boxes(bbox, data.shape[2:4])
        bbox /= fscale
        all_bboxes.append(bbox)
        all_scores.append(scores[0].asnumpy())

        # hflip
        rois, scores, bbox_deltas = net(data[:, :, :, :-1], im_info)
        rois = rois[:, 1:].asnumpy()
        bbox_deltas = bbox_deltas[0].asnumpy()
        # bbox_deltas = pre_compute_deltas(bbox_deltas, bbox_stds=bbox_stds)
        bbox = bbox_pred(rois, bbox_deltas)
        bbox = clip_boxes(bbox, data.shape[2:4])

        tmp = bbox[:, 0::4].copy()
        bbox[:, 0::4] = data.shape[3] - bbox[:, 2::4]  # x0 = w - x0
        bbox[:, 2::4] = data.shape[3] - tmp  # x1 = w -x1
        bbox /= fscale

        all_bboxes.append(bbox)
        all_scores.append(scores[0].asnumpy())


    all_bboxes = np.concatenate(all_bboxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    pred_bboxes = []
    pred_scores = []
    pred_clsid = []
    for j in range(1, all_scores.shape[1]):
        cls_scores = all_scores[:, j, np.newaxis]
        cls_boxes = all_bboxes[:, 4:8] if config.CLASS_AGNOSTIC else all_bboxes[:, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores))
        keep = nms_wrapper(cls_dets.astype('f'))
        cls_dets = cls_dets[keep, :]
        cls_dets = cls_dets[cls_dets[:, -1] > threshold, :]
        pred_bboxes.append(cls_dets[:, :4])
        pred_scores.append(cls_dets[:, 4])
        pred_clsid.append(j * np.ones(shape=(cls_dets.shape[0],), dtype=np.int))
    pred_bboxes = np.concatenate(pred_bboxes, axis=0)
    pred_scores = np.concatenate(pred_scores, axis=0)
    pred_clsid = np.concatenate(pred_clsid, axis=0)
    if viz:
        import gluoncv
        import matplotlib.pyplot as plt
        gluoncv.utils.viz.plot_bbox(img_ori[:, :, ::-1], bboxes=pred_bboxes, scores=pred_scores, labels=pred_clsid,
                                    thresh=.5)
        plt.show()
    return pred_bboxes, pred_scores, pred_clsid





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lun', help='number of lun(a or b)', default='A')
    opt = parser.parse_args()
    if opt.lun=='A':
        data_path = "/data1/bupi_data/guangdong1_round1_testA_20190818/"
    elif opt.lun=='B':
        data_path = '/home/dell/DaringTang/data/'
    import os, tqdm
    print('start test on ', opt.lun, ' dataset')
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    gpu_id = 2
    ctx = [mx.gpu(gpu_id)]
    update_config(
        "/home/dell/DaringTang/works/Detection/code/DCN/fpn/"
        "resnet_v1_101_coco_trainval_fpn_dcn_end2end_ohem.yaml")
    sym = get_symbol(config)
    net = SymbolBlock(sym=sym, input_names=["data", "im_info"],
                      resume = "/home/dell/DaringTang/works/Detection/code/submit1/"
                               "weights-fpn-dcn-6-1000.params"                     # 铝型材料训练好的模型
                      )
    net.collect_params().reset_ctx(ctx)
    im_names = list(lsdir(data_path, suffix=".jpg"))
    # im_names = im_names[:10]

    nms_wrapper = nms = gpu_nms_wrapper(config.TEST.NMS, gpu_id)
    results = []
    for im_name in tqdm.tqdm(im_names):
        TEST_SCALES = [[512, 1280]]
        bboxes, scores, labels = im_detect_bbox_aug(net, nms_wrapper, im_name, TEST_SCALES,
                                                    config.network.PIXEL_MEANS,
                                                    config.TRAIN.BBOX_STDS,
                                                    ctx=ctx, viz=False)
        import matplotlib.pyplot as plt

        plt.show()
        for bbox, score, label in zip(bboxes, scores, labels):
            one_img = {}

            one_img["name"] = os.path.basename(im_name)
            # print(one_img["filename"])

            one_img["category"] = label
            # print(label)

            for i in range(4):
                bbox[i] = '%.2f'%bbox[i]
            one_img["bbox"] = list(bbox[:4])
            # print('one_img["bbox"]', one_img["bbox"])

            one_img["score"] = score
            # print('score', score)
            results.append(one_img)
    import json

    json.dump(results, open('/home/dell/DaringTang/works/Detection/code/result/'
                            'bupi_'+opt.lun+'_results_DCN_9_2.json', "wt"))
