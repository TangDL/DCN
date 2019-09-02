import os.path

import numpy as np
from PIL import Image, ImageDraw
import json
from tqdm import tqdm

# ----------------------------------------------------- PARAMETERS -----------------------------------------------------
image_path = "/media/dell/668e232d-4bee-4d33-8dd9-fe6b54b738d3/data/rssrai2019_object_detection/test/images/"
save_path = "/home/dell/GZX/trained_model/output/"
json_path = '/home/dell/GZX/results_DCN.json'
class_name = [u"large-vehicle", u"swimming-pool", u"helicopter", u"bridge", u"plane", u"ship",
            u"soccer-ball-field", u'basketball-court', u'airport', u'container-crane', u"ground-track-field",
            u"small-vehicle", u"harbor", u"baseball-diamond", u"tennis-court", u"roundabout",
            u"storage-tank", u"helipad"]
# ----------------------------------------------------------------------------------------------------------------------

with open(json_path, 'rt') as f:
    info = json.load(f)

output_result = info['results']
img_num = len(output_result)

for i in tqdm(range(img_num)):
    img = Image.open(image_path+output_result[i]['filename'])
    draw = ImageDraw.Draw(img)
    rects = output_result[i]['rects']
    for j in range(rects.__len__()):
        score = str(round(rects[j]['confidence'], 2))
        obj = class_name[rects[j]['label']]
        text = obj +":"+score
        xmin = rects[j]['xmin']
        ymin = rects[j]['ymin']
        xmax = rects[j]['xmax']
        ymax = rects[j]['ymax']
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
        draw.text((xmin, ymax), text, fill="#ff0000")
    img.save(save_path + output_result[i]['filename'])