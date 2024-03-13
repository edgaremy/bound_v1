import os

def get_dataset_size(dataset_folder, one_class_only=False):
    train_folder = os.path.join(dataset_folder, 'train')
    test_folder = os.path.join(dataset_folder, 'test')
    val_folder = os.path.join(dataset_folder, 'val')

    if one_class_only:
        print("For detection only (1 Class)")
    else:
        # Get the number of classes
        classes = os.listdir(train_folder)
        num_classes = len(classes)
        print('Number of classes:', num_classes)

    # Get the number of images in train, test, and val folders
    num_train_images = sum(len(files) for _, _, files in os.walk(train_folder))
    num_test_images = sum(len(files) for _, _, files in os.walk(test_folder))
    num_val_images = sum(len(files) for _, _, files in os.walk(val_folder))

    # Calculate the total number of images
    total_images = num_train_images + num_test_images + num_val_images

    print('Number of train images:', num_train_images)
    print('Number of test images:', num_test_images)
    print('Number of val images:', num_val_images)
    print('Total number of images:', total_images)

# Example Usage:
print("Dataset Bee BD307:")
dataset_folder = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset"
get_dataset_size(dataset_folder)

print("\nDataset Bee BD71:")
dataset_folder = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_visited/images"
get_dataset_size(dataset_folder, one_class_only=True)

print("\nDataset Bee BD71 Cropped:")
dataset_folder = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/Cropped_BD_71_True/images"
get_dataset_size(dataset_folder)

print("\nDataset Bee BD1")
dataset_folder = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_1class/images"
get_dataset_size(dataset_folder, one_class_only=True)

print("\nDataset Arthropods LIMIT6")
dataset_folder = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT6/dataset/images"
get_dataset_size(dataset_folder, one_class_only=True)

print("\nDataset Lepinoc")
dataset_folder = "/mnt/disk1/datasets/Lepinoc_2022/splitted_dataset/Task_Lepinoc/images"
get_dataset_size(dataset_folder, one_class_only=True)

