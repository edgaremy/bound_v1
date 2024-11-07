import os
import glob
from tqdm import tqdm


def get_bbox_stats(dataset_folder):
    labels_folder = os.path.join(dataset_folder, 'labels')
    label_files = glob.glob(os.path.join(labels_folder, 'train', '*.txt')) + \
                  glob.glob(os.path.join(labels_folder, 'val', '*.txt')) + \
                  glob.glob(os.path.join(labels_folder, 'test', '*.txt'))
    total_bboxes = 0
    total_bbox_area = 0
    total_images = len(label_files)

    for label_file in tqdm(label_files):

        with open(label_file, 'r') as f:
            bboxes = f.readlines()
            total_bboxes += len(bboxes)
            for bbox in bboxes:
                _, x_center, y_center, width, height = map(float, bbox.split())
                bbox_area = width * height
                total_bbox_area += bbox_area # bbox_area is already in percentage of the image area with yolo format

    mean_bbox_size = total_bbox_area / total_bboxes if total_bboxes else 0
    mean_bboxes_per_image = total_bboxes / total_images if total_images else 0

    print(f"\nDataset: {dataset_folder}")
    print(f"{total_bboxes} bounding boxes for {total_images} images")
    print(f"Mean bounding box size: {mean_bbox_size:.2f}%")
    print(f"Mean bounding box number per image: {mean_bboxes_per_image:.2f}")

    return mean_bbox_size, mean_bboxes_per_image

# Example usage:
dataset_list = ["/mnt/disk1/datasets/iNaturalist/Arthropods/dataset17"]
for dataset_folder in dataset_list:
    get_bbox_stats(dataset_folder)
