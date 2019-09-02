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

import _init_paths

import cv2
import yaml
import argparse
import pprint
import os
import sys
from config.config import config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster-RCNN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
print(curr_path)
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))
sys.path.append('/home/dell/DaringTang/works/Detection/code1/DCN/lib/')
sys.path.append("..")

import shutil
import numpy as np
import mxnet as mx


from symbols import *
from core.loader import PyramidAnchorIterator
from core import callback, metric
from core.module import MutableModule
from lib.utils.create_logger import create_logger
from lib.utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from lib.utils.load_model import load_param
from lib.utils.PrefetchingIter import PrefetchingIter
from lib.utils.lr_scheduler import WarmupMultiFactorScheduler


class StepSaveCkpt(object):

    def __init__(self, mod, mean, std, frequent=1000):
        self.frequent = frequent
        # self.prefix = "/home/dell/DaringTang/works/Detection/code/submit/model/weights-fpn-dcn-{}-{}.params"   # 模型存储路径
        self.prefix = "/home/dell/DaringTang/works/Detection/code/model/submit/weights-fpn-dcn-{}-{}.params"  # 模型存储路径
        self.mean = mean                                         # 图片像素均值
        self.std = std
        self.mod = mod

    def __call__(self, param):
        means = self.mean
        stds = self.std
        count = param.nbatch                                     # 每1轮迭代使用的样本量
        epoch = param.epoch
        if count % self.frequent == 0:
            arg, aux = self.mod.get_params()
            arg['bbox_pred_weight_test'] = (arg['bbox_pred_weight'].T * mx.nd.array(stds)).T
            arg['bbox_pred_bias_test'] = arg['bbox_pred_bias'] * mx.nd.array(stds) + mx.nd.array(means)
            save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in arg.items()}
            save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux.items()})
            save_path = self.prefix.format(epoch, count)         # 定义存储路径
            mx.nd.save(save_path, save_dict)
            self.mod.logger.info("Saved ckpt to {}.".format(save_path))
            arg.pop('bbox_pred_weight_test')                     # pop删除列表中的元素，默认删除最后一个
            arg.pop('bbox_pred_bias_test')

def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    mx.random.seed(3)                                            # 生成随机数种子
    np.random.seed(3)
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)  # 创建日志及输出
    prefix = os.path.join(final_output_path, prefix)                                                   # 输出模型路径
    print(prefix)

    # load symbol
    shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), final_output_path)        # 拷贝文件和状态信息
    sym_instance = eval(config.symbol + '.' + config.symbol)()                                        # 定义resnet_v1_101_fpn_dcn_rcnn类
    sym = sym_instance.get_symbol(config, is_train=True)                                              # 返回一个生成的symbol,即要用来训练的网络

    feat_pyramid_level = np.log2(config.network.RPN_FEAT_STRIDE).astype(int)                          # log2取对数
    feat_sym = [sym.get_internals()['rpn_cls_score_p' + str(x) + '_output'] for x in feat_pyramid_level]

    # setup multi-gpu
    batch_size = len(ctx)                                                                             # 基于GPU的数量设定batch的大小
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(config)                                                                             # 打印配置信息，并将配置信息放入日志
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in config.dataset.image_set.split('+')]                               # 确定数据集名称
    # dir1 ="/home/dell/DaringTang/data/train_round2.roidb"
    roidbs = [load_gt_roidb(config.dataset.dataset, image_set,
                            config.dataset.root_path, config.dataset.dataset_path, flip=config.TRAIN.FLIP)                                                   # 加载roi信息，增加翻转后的标注信息
              for image_set in image_sets]
    roidb = merge_roidb(roidbs)                                                                       # roidb are list, concat them together
    roidb = filter_roidb(roidb, config)                                                               # 过滤无用的roi信息，重合度小于阈值时删去
    # load training data
    train_data = PyramidAnchorIterator(feat_sym, roidb, config, batch_size=input_batch_size, shuffle=config.TRAIN.SHUFFLE,
                                       ctx=ctx, feat_strides=config.network.RPN_FEAT_STRIDE, anchor_scales=config.network.ANCHOR_SCALES,
                                       anchor_ratios=config.network.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING,
                                       allowed_border=np.inf)                                        # This Iter will provide roi data to Fast R-CNN network
    # load val data
    # dir2="/home/dell/DaringTang/data/train_round2_val.roidb"
    # roidbs_val = [load_gt_roidb(config.dataset.dataset, image_set, config.dataset.root_path, config.dataset.dataset_path
    #                         ,roidb_dir=dir2, flip=config.TRAIN.FLIP)]
    # roidb_val = merge_roidb(roidbs_val)
    # roidb_val = filter_roidb(roidb_val, config)
    # val_data = PyramidAnchorIterator(feat_sym, roidb_val, config, batch_size=input_batch_size, shuffle=config.TRAIN.SHUFFLE,
    #                                    ctx=ctx, feat_strides=config.network.RPN_FEAT_STRIDE, anchor_scales=config.network.ANCHOR_SCALES,
    #                                    anchor_ratios=config.network.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING,
    #                                    allowed_border=np.inf)

    # infer max shape
    max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (config.TRAIN.BATCH_IMAGES, 100, 5)))                         # 返回数据和标签的最大数量
    print 'providing maximum shape', max_data_shape, max_label_shape

    data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)         # create a dict
    pprint.pprint(data_shape_dict)                                                                   # 打印data和label信息
    sym_instance.infer_shape(data_shape_dict)

    # load and initialize params
    import glob
    # if len(glob.glob('/home/dell/DaringTang/works/Detection/code/submit1/*.params'))!= 0:   # 判断是否断点开始
    #     config.TRAIN.RESUME = True
    if config.TRAIN.RESUME:
        print('continue training from ', begin_epoch)
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        print('load from bupi pretrained model')                                                # 一般是从这里开始
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)                  # 载入模型的参数名和参数值
        sym_instance.init_weight(config, arg_params, aux_params)

    # check parameter shapes
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)              # 检查模型参数是否对应

    # create solver
    fixed_param_prefix = config.network.FIXED_PARAMS                                          # 载入模型层次
    data_names = [k[0] for k in train_data.provide_data_single]                               # ['data', 'im_info', 'gt_boxes']
    label_names = [k[0] for k in train_data.provide_label_single]                             # ['label', 'bbox_target', 'bbox_weight']

    mod = MutableModule(sym, data_names=data_names, label_names=label_names,                  # 定义模型
                        logger=logger, context=ctx, max_data_shapes=[max_data_shape for _ in range(batch_size)],
                        max_label_shapes=[max_label_shape for _ in range(batch_size)], fixed_param_prefix=fixed_param_prefix)

    if config.TRAIN.RESUME:
        mod._preload_opt_states = '%s-%04d.states'%(prefix, begin_epoch)

    # decide training params
    # metric
    rpn_eval_metric = metric.RPNAccMetric()                                                    # RPN网络的分类准确率
    rpn_cls_metric = metric.RPNLogLossMetric()                                                 # RPN网络的分类损失
    rpn_bbox_metric = metric.RPNL1LossMetric()                                                 # RPN网络的回归损失
    rpn_fg_metric = metric.RPNFGFraction(config)                                               # RPN网络的前景识别率
    eval_metric = metric.RCNNAccMetric(config)                                                 # RPN网络的前景识别正确率
    eval_fg_metric = metric.RCNNFGAccuracy(config)                                             # RCNN的检测准确率
    cls_metric = metric.RCNNLogLossMetric(config)                                              # RCNN的分类损失
    bbox_metric = metric.RCNNL1LossMetric(config)                                              # RCNN的回归损失
    eval_metrics = mx.metric.CompositeEvalMetric()                                             # 用这个类调用上面的metric
    # rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric
    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, rpn_fg_metric, eval_fg_metric, eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)                                                         # 将child_metric通过CompositeEvalMetric()中的方法加入到eval_metric中
    # callback
    means = np.tile(np.array(config.TRAIN.BBOX_MEANS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)    # 初始化每个类别的均值
    stds = np.tile(np.array(config.TRAIN.BBOX_STDS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)      # 初始化标准差 ？
    batch_end_callback = [callback.Speedometer(train_data.batch_size, frequent=args.frequent), StepSaveCkpt(mod, means, stds)]    # 显示batch进度，存储每一个参数
    # eval_end_callback = [callback.Speedometer(val_data.batch_size, frequent=args.frequent), StepSaveCkpt(mod, means, stds)]   # 显示每一个验证的性能

    epoch_end_callback = [mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True), callback.do_checkpoint(prefix, means, stds)] # 保存模型
    # decide learning rate
    base_lr = lr
    lr_factor = config.TRAIN.lr_factor                                                      # 在cfg中没有找到这个参数lr_factor
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]                               # what's this param mean ?
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]      # 4,6 epoch that need to change lr
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)

    # Reduce learning rate in factor at steps specified in a list
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr, config.TRAIN.warmup_step)

    # optimizer
    optimizer_params = {'momentum': config.TRAIN.momentum,
                        'wd': config.TRAIN.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'clip_gradient': None}
    #
    if not isinstance(train_data, PrefetchingIter):
        train_data = PrefetchingIter(train_data)
    # if not isinstance(val_data, PrefetchingIter):
    #     val_data = PrefetchingIter(val_data)

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)


def main():
    print('Called with argument:', args)
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]          # 训练用的GPU
    # check if there is file in model folder
    import glob
    # params_files = glob.glob('/root/project/submit/model/dcn_trained/resnet_v1_101_coco_trainval_fpn_dcn_end2end_ohem/train2014/*.params')
    # params_files = glob.glob(
    #     '/home/dell/DaringTang/works/Detection/code/submit1/*.params')                                                # check if resume
    # get the number of params files
    # if len(params_files)!=0:
    #         config.network.pretrained_epoch = len(params_files)
    #         config.TRAIN.begin_epoch = len(params_files)
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_epoch, config.TRAIN.model_prefix,
              config.TRAIN.begin_epoch, config.TRAIN.end_epoch, config.TRAIN.lr, config.TRAIN.lr_step)

if __name__ == '__main__':
    main()
