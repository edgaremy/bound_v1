import os
import csv
from PIL import Image

import convert_bbox_format as bbox_converter

import csv_manipulation.get_hierarchy as hierarchy

def convert_labels_to_csv(classes_path, labels_path, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_path', 'xmin', 'xmax', 'ymin', 'ymax', 'label', 'image_width', 'image_height', 'class', 'order', 'family', 'genus', 'species'])
        
        unfound_classes = 0

        # Iterate over each class folder
        for folder in os.listdir(classes_path):
        
            images_path = os.path.join(classes_path, folder)
            for image in os.listdir(images_path):
                    image_name = os.path.splitext(image)[0]
                    image_path = os.path.join(images_path, image)
                    txt_path = os.path.join(labels_path, image_name + '.txt')
                    
                    # Get hierarchy
                    # taxon_id = image_name.split('_')[0]
                    # hierarchy_list = hierarchy.get_hierarchy_from_taxon_id(taxon_id)
                    try:
                        hierarchy_list = hierarchy.get_hierarchy_from_name(folder)
                    except:
                        hierarchy_list = ['', '', '', '', folder]
                        unfound_classes += 1
                    class_, order, family, genus, species = hierarchy_list

                    if os.path.exists(txt_path):
                        with open(txt_path, 'r') as f:
                            for line in f:
                                line = line.strip().split()
                                line = [float(x) for x in line]

                                image_width, image_height = get_image_dimensions(image_path)
                                # label, x, y, w, h = line
                                xmin, ymin, xmax, ymax = bbox_converter.get_bbox_from_yolo_format(image_width, image_height, line[1:5])
                                label = 'Arthropod'

                                writer.writerow([image, xmin, xmax, ymin, ymax, label, image_width, image_height, class_, order, family, genus, species])
                    else:
                        writer.writerow([image, '', '', '', '', '', '', '', class_, order, family, genus, species])

        print(f'Number of unfound classes in iNat: {unfound_classes}')

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        image_width, image_height = img.size
    return image_width, image_height


images_path = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/DG'
labels_path = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/DG_evaluate_predictions/DG_predicted_labels'
csv_file = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/DG_evaluate_predictions/DG_predicted_labels.csv'
convert_labels_to_csv(images_path, labels_path, csv_file)
