import os
import csv
from PIL import Image
import json

import pandas as pd



source_path = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71/'
img_dst_path = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_visited/classes/'
label_dst_path = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_visited/labels/default/'


json_path = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_True_Annotations/vérifiés/'


with open('/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_True_Annotations/liste_classes_71.csv', newline='') as csvfile:
	filereader = csv.DictReader(csvfile, delimiter='\n')
	class_idx = 0
	for row in filereader:
		specie = row['species']
		print(specie)

		# Opening JSON file
		f = open(json_path + specie + '.json')
		  
		# returns JSON object as 
		# a dictionary
		data = json.load(f)

		if not os.path.exists(img_dst_path + specie):
			os.mkdir(img_dst_path + specie)
		# if not os.path.exists(label_dst_path + specie):
		# 	os.mkdir(label_dst_path + specie)

		for img in data:
		
			if img['visited']==1:
			
				img_path = img['file_path'][8:]
				
				aux = img_path.split('.')
				img_extension = aux[-1]
				img_name = img_path.split('/')[-1]
				
				
				boxes = img['boxes']
				img_full_path = source_path + img_path

				if os.path.exists(img_full_path) and os.path.getsize(img_full_path) > 0:

					im = Image.open(img_full_path)
					w, h = im.size
					
					box_list = []

					for box in boxes:
						x_center = (box['xmin'] + box['xmax']) / 2
						y_center = (box['ymin'] + box['ymax']) / 2
						width = abs(box['xmax'] - box['xmin'])
						height = abs(box['ymax'] - box['ymin'])

						box_list.append([class_idx, x_center, y_center, width, height])

					if len(box_list) > 0:
						im.save(img_dst_path + img_path)

						df = pd.DataFrame(box_list, columns=['class', 'x1', 'y1', 'w', 'h'])
						df.to_csv(label_dst_path + img_name.split('.')[0] + '.txt', index=False, header=False, sep=' ', float_format='%.6f')
				
				else:
					print("1 missing image")
					

		f.close()
		class_idx += 1
