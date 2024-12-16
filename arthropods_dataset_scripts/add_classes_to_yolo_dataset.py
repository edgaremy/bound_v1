import os
import shutil
import pandas as pd

def get_taxon_at_level(hierarchy_df, image_name, level='order'):
    taxon_id = image_name.split('_')[0]
    row = hierarchy_df[hierarchy_df['taxon_id'] == int(taxon_id)].iloc[0]
    
    return row[level]

def copy_and_update_yolo_dataset(input_dataset, output_folder, hierarchy_csv, taxon_level='order'):

    hierarchy_df = pd.read_csv(hierarchy_csv)

    # First get all the classes at the desired taxon level
    classes = set()
    for split in ['train', 'val', 'test']:
        input_labels_path = os.path.join(input_dataset, 'images', split)
        for image_name in os.listdir(input_labels_path):
            classes.add(get_taxon_at_level(hierarchy_df, image_name, taxon_level))
    # Order the classes
    classes = sorted(list(classes))
    # Create a dictionary to map the classes to their index
    class_to_index = {class_: i for i, class_ in enumerate(classes)}

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for split in ['train', 'val', 'test']:
        input_images_path = os.path.join(input_dataset, 'images', split)
        input_labels_path = os.path.join(input_dataset, 'labels', split)
        output_images_path = os.path.join(output_folder, 'images', split)
        output_labels_path = os.path.join(output_folder, 'labels', split)
        
        if not os.path.exists(output_images_path):
            os.makedirs(output_images_path)
        if not os.path.exists(output_labels_path):
            os.makedirs(output_labels_path)
        
        for image_name in os.listdir(input_images_path):
            input_image_file = os.path.join(input_images_path, image_name)
            input_label_file = os.path.join(input_labels_path, os.path.splitext(image_name)[0] + '.txt')
            output_image_file = os.path.join(output_images_path, image_name)
            output_label_file = os.path.join(output_labels_path, os.path.splitext(image_name)[0] + '.txt')
            
            # Copy image file
            shutil.copy(input_image_file, output_image_file)
            
            # Copy associated label file, update class index for each line if the label file exists:
            if os.path.exists(input_label_file):
                with open(input_label_file, 'r') as f_in, open(output_label_file, 'w') as f_out:
                    for line in f_in:
                        class_, x, y, w, h = line.split()
                        class_index = class_to_index[get_taxon_at_level(hierarchy_df, image_name, taxon_level)]
                        f_out.write(f'{class_index} {x} {y} {w} {h}\n')
    
    # Create the yaml file
    with open(os.path.join(output_folder, 'dataset.yaml'), 'w') as f:
        f.write(f'path: {output_folder}\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('test: images/test\n\n')
        f.write(f'nc: {len(classes)}\n\n')
        f.write('names:\n')
        for i, class_ in enumerate(classes):
            f.write(f'  {i}: "{class_}"\n')
               


# Example usage
input_dataset = '/mnt/disk1/datasets/iNaturalist/Arthropods/FINAL_DATASET'
output_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/tmp2'
hierarchy_csv = 'dataset_analysis/arthro_dataset_hierarchy.csv'
taxon_level = 'order'
copy_and_update_yolo_dataset(input_dataset, output_folder, hierarchy_csv, taxon_level)
