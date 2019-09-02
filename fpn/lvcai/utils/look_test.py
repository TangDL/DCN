#encoding=utf-8
import cv2
import matplotlib.pyplot as plt
import json
import os
import pickle

val_data = json.load(open('/data3/zyx/project/al_detect/treeBoyDefect-master/Deformable-ConvNets/fpn/lvcai/result/round2_test_b/fused_0_after.json'))
img_root = u'/data1/zyx/cltdevelop/lvcai/guangdong_round2_test_b_20181106'
for index, one_result in enumerate(val_data[u'results']):
    img = cv2.imread(os.path.join(img_root, one_result[u'filename']))
    for one_rect in one_result[u'rects']:
        if one_rect[u'confidence']<0.5:
            continue
        img = cv2.rectangle(img,(one_rect[u'xmin'], one_rect[u'ymin']) ,(one_rect[u'xmax'], one_rect[u'ymax']), (255, 0, 0), 5)
        img = cv2.putText(img,one_rect[u'label'], (one_rect[u'xmin'], one_rect[u'ymin']), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 2)
    print(one_result[u'filename'])
    plt.figure()
    plt.imshow(img[:, :, ::-1])
    plt.show()
print(val_data)