# encoding=utf-8
# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Haozhi Qi
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import mxnet as mx
import numpy as np


# 这部分是获取RPN网络的变量名，因为RPN网络有多个输出，所以要根据变量名来读取对应的输出。
def get_rpn_names():
    pred = ['rpn_cls_prob', 'rpn_bbox_loss']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


# 这部分是获取检测网络（Fast RCNN）的变量名，因为检测网络有多个输出，所以要根据变量名来读取对应的输出。
def get_rcnn_names(cfg):
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    if cfg.TRAIN.ENABLE_OHEM or cfg.TRAIN.END2END:
        pred.append('rcnn_label')
    if cfg.TRAIN.END2END:
        rpn_pred, rpn_label = get_rpn_names()
        pred = rpn_pred + pred
        label = rpn_label
    return pred, label


# RPN前景预测准确率
class RCNNFGAccuracy(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNFGAccuracy, self).__init__('R-CNN FG Accuracy')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]
        num_classes = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, num_classes).argmax(axis=1).astype('int32')
        # selection of ground truth label is different from softmax or sigmoid classifier
        label = label.asnumpy().reshape(-1, ).astype('int32')
        keep_inds = np.where(label > 0)
        # filter out -1 label because of OHEM or invalid samples
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(np.equal(pred_label.flat, label.flat))
        self.num_inst += pred_label.shape[0]


# RPN预测框中类前景的概率
class RPNFGFraction(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RPNFGFraction, self).__init__('Proposal FG Fraction')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]
        num_classes = pred.shape[-1]
        # selection of ground truth label is different from softmax or sigmoid classifier
        label = label.asnumpy().reshape(-1, ).astype('int32')
        fg_inds = np.where(label > 0)[0]                              # 前景数量
        bg_inds = np.where(label == 0)[0]                             # 后景数量
        self.sum_metric += fg_inds.shape[0]
        self.num_inst += (fg_inds.shape[0] + bg_inds.shape[0])


# RPNAccMetric类是RPN网络的分类准确率计算。对于一张大小为600*800的输入图像而言，
# pred是1*2*342*50，label是1*17100(342*50)
class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')             # 类继承
        self.pred, self.label = get_rpn_names()

    # preds是包含多个输出的列表，所以通过index进行索引，labels同理。
    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)                         #  获取pred和label信息
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))                 # 降成二维数组
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds                                  # 类别准确率的计算不包含无效标签。
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)    # 正确预测的个数
        self.num_inst += len(pred_label.flat)                       # 总个数


# RCNNAccMetric类是检测网络的分类准确率计算。
class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


# RPNLogLossMetric类是用来计算RPN网络的分类损失。
class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


# RCNNLogLossMetric类是用来计算检测网络的分类损失。
class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


# RPNL1LossMetric类是用来计算RPN网络的回归损失。
class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        label = labels[self.label.index('rpn_label')].asnumpy()
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


# RCNNL1LossMetric类是用来计算检测网络的回归损失。
class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        if self.ohem:
            label = preds[self.pred.index('rcnn_label')].asnumpy()
        else:
            if self.e2e:
                label = preds[self.pred.index('rcnn_label')].asnumpy()
            else:
                label = labels[self.label.index('rcnn_label')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst
