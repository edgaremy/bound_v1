import os
import json
import shutil
import argparse

def filter_coco_json(image_folder, json_file, output_json_file):
    # Get the list of image filenames in the folder
    image_filenames = set(os.listdir(image_folder))
    
    # Load the original COCO JSON file
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # Filter images and annotations
    filtered_images = []
    image_id_map = {}
    for image in coco_data['images']:
        if image['file_name'] in image_filenames:
            filtered_images.append(image)
            image_id_map[image['id']] = image['file_name']
    
    filtered_annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] in image_id_map:
            if ann['category_id'] == 0:
                ann['category_id'] = 1
            filtered_annotations.append(ann)
    
    # Create the new COCO JSON structure
    # Ensure the first category's id is 1 and not 0
    for category in coco_data['categories']:
        if category['id'] == 0:
            category['id'] = 1

    filtered_coco_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': coco_data['categories']
    }
    
    # Save the filtered COCO JSON to a new file
    with open(output_json_file, 'w') as f:
        json.dump(filtered_coco_data, f, indent=4)

# Example usage
folder_path = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT1/tmp/data_short'
json_file = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT1/tmp/labels.json'
output_json_file = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT1/tmp/labels_filtered.json'

filter_coco_json(folder_path, json_file, output_json_file)