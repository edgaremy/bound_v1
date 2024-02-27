import os
import json
from tqdm import tqdm
# from xml.dom import minidom
# import csv  
# import numpy as np
# import random
# import cv2
# import matplotlib.pyplot as plt
# import wget
# import pickle
# import random
# from PIL import Image
# from operator import itemgetter

import convert_bbox_format as bbox_converter


json_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT5/dataset(prediction)/json_validated'
#image_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT1/dataset(prediction)/images'
labels_output_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT5/dataset(prediction)/labels_validated'


os.makedirs(labels_output_folder, exist_ok=True)

json_files = os.listdir(json_folder)
print(json_files)

for json_file in json_files:
    print(json_file[:-5])
     
	# with open('CSV_v/' + json_file[:-5] + '.csv', 'w', encoding='UTF8', newline='') as f:
	# 	writer = csv.writer(f)
    with open(os.path.join(json_folder, json_file), 'r') as f:
        data = json.load(f)
        for d in tqdm(data):
            if d['visited'] == 1 and len(d['boxes']) > 0:
                img_name = d['File_path']
                img_name = img_name.split('/')[-1]
            
                w = d['width']
                h = d['height']

                boxes = d['boxes']
                for box in boxes:
                    
                    xmin = box['xmin']*w
                    xmax = box['xmax']*w
                    ymin = box['ymin']*h
                    ymax = box['ymax']*h
                    
                    x, y, w, h = bbox_converter.get_yolo_format_from_bbox(w, h, [xmin, ymin, xmax, ymax])
                    label = 0
                    label_file = os.path.join(labels_output_folder, img_name.split('.')[0] + '.txt')
                    
                    with open(label_file, 'a') as f:
                        f.write(f"{label} {x} {y} {w} {h}\n")
                        
                        # try:
                        #     classe = box['specie']
                        #     writer.writerow([img_name, xmin, ymin, xmax, ymax, classe, w, h])
                        #     break
                        # except:
                        #     pass
						