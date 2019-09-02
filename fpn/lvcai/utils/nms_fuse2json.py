import mxnet as mx
import json
import pandas as pd
from tqdm import tqdm
import time
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--lun', help='number of lun(a or b)', default='A')
opt = parser.parse_args()
if opt.lun=='A':
    result1 = '/root/project/code/DCN/fpn/lvcai/result/2_Aresults_DCN.json'
    result2 = '/root/project/code/PANet/tools/2_Aresults_Panet.json'
if opt.lun=='B':
    result1 = '/root/project/code/DCN/fpn/lvcai/result/2_Bresults_DCN.json'
    result2 = '/root/project/code/PANet/tools/2_Bresults_Panet.json'
print('fusing.......')
old_results_file = [result1, result2]

all_dict = []
for old_result in old_results_file:
    all_dict.append(json.load(open(old_result)))
filenames = []
labels = []
scores = []
xmins = []; ymins = []; xmaxs = []; ymaxs = []
json_files = []
# generate dataframe with all boxes
all_result = pd.DataFrame(columns=['file', 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax','json_file'])
json_file = 0
for one_dict in all_dict:
    for one_pic in one_dict['results']:
        pic_name = one_pic['filename']
        for one_rect in one_pic['rects']:
            filenames.append(pic_name)
            labels.append(one_rect['label'].replace('defect',''))
            scores.append(one_rect['confidence'])
            xmins.append(one_rect['xmin'])
            xmaxs.append(one_rect['xmax'])
            ymins.append(one_rect['ymin'])
            ymaxs.append(one_rect['ymax'])
            json_files.append(json_file)
    json_file += 1
all_result['file'] = filenames
all_result['label'] = labels
all_result['score'] = scores
all_result['xmin'] = xmins
all_result['ymin'] = ymins
all_result['xmax'] = xmaxs
all_result['ymax'] = ymaxs
all_result['json_file'] = json_files

#delete boxes using nms
all_pics = set(all_result['file'].values)
new_files = []; new_labels = []; new_scores = []; new_xmins = [];
new_ymins = []; new_xmaxs = []; new_ymaxs = []
for pic in tqdm(all_pics):
    time0 = time.clock()
    # print(pic)
    boxes = []
    boxes_tmp = []
    match_df = all_result[all_result['file']==pic]
    for index, row in match_df.iterrows():
        boxes_tmp.append([int(row['label']), row['score'], row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['json_file']])
        boxes.append([int(row['label']), row['score'], row['xmin'], row['ymin'], row['xmax'], row['ymax']])
    boxes = mx.nd.array(boxes)
    time1 = time.clock()
    boxes_tmp = mx.nd.array(boxes_tmp)
    nms_boxes = mx.nd.contrib.box_nms(boxes,overlap_thresh=0.3,force_suppress=False,out_format='corner', id_index=0)
    time2 = time.clock()
    # print('time1:', time1-time0)
    # print('time2:', time2-time1)
    for one_box in nms_boxes:
        if one_box[0]<0:
            continue
        new_files.append(pic); new_labels.append(int(one_box[0].asnumpy()));
        new_scores.append(float(one_box[1].asnumpy()))
        new_xmins.append(int(one_box[2].asnumpy())); new_ymins.append(int(one_box[3].asnumpy()));
        new_xmaxs.append(int(one_box[4].asnumpy())); new_ymaxs.append(int(one_box[5].asnumpy()));
new_result = pd.DataFrame({'file': new_files,
                           'label': new_labels,
                           'score': new_scores,
                           'xmin': new_xmins,
                           'ymin': new_ymins,
                           'xmax': new_xmaxs,
                           'ymax': new_ymaxs})

# generate new json file
print('generate new json file')
old_json = json.load(open(result1))
result_df = new_result
results = {}
results["results"] = []
for i in range(len(old_json['results'])):
    one_img = {}
    one_img['filename'] = old_json['results'][i]['filename']
    one_img["rects"] = []
    match_df = result_df[result_df['file']==one_img['filename']]
    if match_df.shape[0] != 0:
        for index, row in match_df.iterrows():

            one_rect = {}
            one_rect["xmin"] = row['xmin']
            one_rect["ymin"] = row['ymin']
            one_rect["xmax"] = row['xmax']
            one_rect["ymax"] = row['ymax']
            one_rect["confidence"] = row['score']
            one_rect["label"] = "defect%d" % (row['label'])
            one_img["rects"].append(one_rect)
    else:
        print("no detection", one_img['filename'])
    results["results"].append(one_img)
os.remove(result1)
os.remove(result2)
json.dump(results, open('/root/project/submit/2_'+opt.lun+'results.json', "wt"))