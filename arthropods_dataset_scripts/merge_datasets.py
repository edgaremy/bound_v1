import os
import shutil

def merge_datasets(source_dataset, target_dataset):
    # Get the source and target paths
    source_labels_path = os.path.join(source_dataset, 'labels')
    source_images_path = os.path.join(source_dataset, 'images')
    target_labels_path = os.path.join(target_dataset, 'labels')
    target_images_path = os.path.join(target_dataset, 'images')

    # Copy the source labels and images to the target dataset
    shutil.copytree(source_labels_path, target_labels_path, dirs_exist_ok=True)
    shutil.copytree(source_images_path, target_images_path, dirs_exist_ok=True)

    # Remove cache for labels if exists:
    cache_path = os.path.join(target_dataset, 'labels', 'train.cache')
    if os.path.exists(cache_path):
        os.remove(cache_path)
    cache_path = os.path.join(target_dataset, 'labels', 'val.cache')
    if os.path.exists(cache_path):
        os.remove(cache_path)
    cache_path = os.path.join(target_dataset, 'labels', 'test.cache')
    if os.path.exists(cache_path):
        os.remove(cache_path)

    print("Dataset merged successfully!")

# Example usage
# source_dataset = '/mnt/disk1/datasets/iNaturalist/Arthropods/TEST2/dataset'
# target_dataset = '/mnt/disk1/datasets/iNaturalist/Arthropods/TEST/dataset'

# merge_datasets(source_dataset, target_dataset)
    
def create_yaml(yaml_path):
    with open(yaml_path, 'w') as f:
        folder_path = os.path.dirname(yaml_path)
        f.write('path: ' + folder_path + '/\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('#test: images/test\n')
        f.write('\n')
        f.write('# number of classes\n')
        f.write('nc: 1\n')
        f.write('\n')
        f.write('# Classes\n')
        f.write('names:\n')
        f.write('  0: "Arthropod"')
        print("YAML file created successfully!")

# Example usage
# yaml_path = '/mnt/disk1/datasets/iNaturalist/Arthropods/TEST/dataset/dataset.yaml'
# create_yaml(yaml_path)