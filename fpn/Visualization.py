import os
import os.path

import numpy as np
import xml.etree.ElementTree as xmlET
from PIL import Image, ImageDraw
import cPickle as pickle
from tqdm import tqdm
import cv2
import json

# ----------------------------------------------------- PARAMETERS -----------------------------------------------------
image_path = "/home/dell/DaringTang/data/guangdong_round2_test_a_20181011/"
save_path = "/home/dell/DaringTang/data/"
json_path = '/home/dell/DaringTang/works/Detection/code/DCN/fpn/lvcai/result/2_Aresults_DCN.json'
score_threshold = 1e-1
class_name = [u"defect0", u"defect1", u"defect2", u"defect3", u"defect4", u"defect5", u"defect6", u"defect7", u"defect8", u"defect9"]
# ----------------------------------------------------------------------------------------------------------------------

with open(json_path, 'rb') as f:
    info = json.load(f)

output_data = info['results']
img_num = len(output_data)
img_num = 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

for i in tqdm(range(len(img_num))):
    img = Image.open(image_path+output_data[i]['filename'])
    draw = ImageDraw.Draw(img)
    rects = output_data[i]['rects']
    for j in range(len(rects)):
        score = str(round(rects[j]['confidence'], 2))
        obj = rects[j]['label']
        text = obj + ":" + score
        xmin = rects[j]['xmin']
        ymin = rects[j]['ymin']
        xmax = rects[j]['xmax']
        ymax = rects[j]['ymax']
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
        draw.text((xmin, ymin), text, fill="#ff0000")
    img.save(save_path + output_data[i]['filename'])
