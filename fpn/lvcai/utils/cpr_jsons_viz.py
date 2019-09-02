# encoding=utf-8
import json
import cv2
import matplotlib.pyplot as plt
import os

json1 = '/root/project/submit/2_Bresults.json'
json2 = '/root/project/code/DCN/fpn/lvcai/result/round2_test_b/fused_0_after.json'
img_root = u'/root/project/data/guangdong_round2_test_b_20181106'

json_data1 = json.load(open(json1))
json_data2 = json.load(open(json2))

for one_result in json_data1[u'results']:
    img_name = one_result[u'filename']
    for other_result in json_data2[u'results']:
        other_name = other_result[u'filename']
        if img_name == other_name:

            break

    img = cv2.imread(os.path.join(img_root, one_result[u'filename']))
    for one_rect in one_result[u'rects']:
        if one_rect[u'confidence']<0.1:
            continue
        img = cv2.rectangle(img,(one_rect[u'xmin'], one_rect[u'ymin']) ,(one_rect[u'xmax'], one_rect[u'ymax']), (255, 0, 0), 5)
        img = cv2.putText(img,one_rect[u'label'], (one_rect[u'xmin'], one_rect[u'ymin']), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 2)
        img = cv2.putText(img, str(one_rect[u'confidence']), (one_rect[u'xmin'], one_rect[u'ymax']),
                          cv2.FONT_HERSHEY_COMPLEX, 3,
                          (0, 0, 255), 2)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img[:, :, ::-1])

    img = cv2.imread(os.path.join(img_root, other_result[u'filename']))
    for one_rect in other_result[u'rects']:
        if one_rect[u'confidence'] < 0.1:
            continue
        img = cv2.rectangle(img, (one_rect[u'xmin'], one_rect[u'ymin']), (one_rect[u'xmax'], one_rect[u'ymax']),
                            (255, 0, 0), 5)
        img = cv2.putText(img, one_rect[u'label'], (one_rect[u'xmin'], one_rect[u'ymin']), cv2.FONT_HERSHEY_COMPLEX, 3,
                          (0, 0, 255), 2)
        img = cv2.putText(img, str(one_rect[u'confidence']), (one_rect[u'xmin'], one_rect[u'ymax']), cv2.FONT_HERSHEY_COMPLEX, 3,
                          (0, 0, 255), 2)
    plt.subplot(1, 2, 2)
    plt.imshow(img[:, :, ::-1])
    plt.show()
